# Analysis

## Requirements

We need an efficient FP16 GEMM kernel for Intel Battlemage (Xe2-HPG) GPU. The key constraints from the instructions are:

1. **Architecture**: 64 work-items (4 subgroups of 16), A in SLM, B from global/L2
2. **Tile size**: 32×64×32 (TILE_M=32, TILE_N=64, TILE_K=32)
3. **DPAS instruction**: `intel_sub_group_f16_f16_matrix_mad_k16(short8 a, int8 b, float8 acc)` - performs 8×16×16 matrix multiply
4. **K=2048 divides evenly by 32**, so no remainder path needed

## Proposed Kernel Structure

**Work-group layout**: 64 WIs = 4 subgroups × 16 WIs per subgroup
- Each subgroup handles a 32×16 portion of the output tile (TILE_M=32, TILE_N=64 / 4 SGs = 16 columns per SG)
- Wait - actually with DPAS producing 8 rows per call, and we need 32 rows, each SG does 4 DPAS calls per K-step for its 16 columns

**DPAS semantics on Xe2** (exec_size=16):
- `intel_sub_group_f16_f16_matrix_mad_k16(short8 a, int8 b, float8 acc)`:
  - A: short8 = 8 rows × 16 k-elements (packed as fp16 pairs in short)
  - B: int8 = 16 k-elements × 16 columns (packed)
  - Result: float8 = 8 rows × 1 column (per WI, 16 WIs = 16 columns)

**K-loop**: 2048/32 = 64 iterations, unrolled 2x = 32 loop iterations

**SLM for A**: 
- A tile: 32 rows × 32 K-elements × 2 bytes = 2048 bytes per buffer
- Double buffering: 4096 bytes total (well within 64KB SLM)
- Stride with padding to avoid bank conflicts: 32+4 = 36 elements per row → 32×36×2 = 2304 bytes per buffer

**B from global**: Each subgroup loads its own B tile (32×16 = 512 elements = 1024 bytes per K-step)

**Optimizations applied**:
1. K-loop 2x unroll with double-buffered SLM
2. No K-remainder path (K divisible by 32)
3. SLM padding for bank conflict avoidance
4. Vectorized SLM reads using intel_sub_group_block_read
5. Register blocking: 4 DPAS calls per SG = 32 rows × 16 cols accumulator

```OCL
// GEMM: C[M,N] = A[M,K] x B[K,N], all row-major FP16
// Launch: GWS = (N/16, M/32), LWS = (4, 1) with reqd_sub_group_size=16
// Effective: 4 subgroups × 16 WIs = 64 WIs per WG
// Each WG computes a 32×64 output tile
// TILE_M=32, TILE_N=64, TILE_K=32
// SLM: A tile with double buffering, B from global memory

#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable

#define TILE_M 32
#define TILE_N 64
#define TILE_K 32
#define SG_SIZE 16
#define NUM_SG 4
#define SLM_STRIDE 36  // 32 + 4 padding to reduce bank conflicts

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(4, 1, 1)))
__kernel void matmul(
    __global const half* A,
    __global const half* B,
    __global half* C,
    const int M,
    const int K,
    const int N)
{
    // Work-group and subgroup identification
    const int wg_n = get_group_id(0);  // which 64-col block
    const int wg_m = get_group_id(1);  // which 32-row block
    const int sg_id = get_sub_group_id();  // 0..3, each handles 16 columns
    const int sg_lid = get_sub_group_local_id();  // 0..15

    // Global offsets
    const int m_start = wg_m * TILE_M;
    const int n_start = wg_n * TILE_N + sg_id * SG_SIZE;

    // SLM for A double buffer
    __local half slm_a[2 * TILE_M * SLM_STRIDE];

    // Accumulators: 32 rows × 16 cols (per SG), stored as 4 blocks of float8
    float8 acc0 = (float8)(0.0f);  // rows 0-7
    float8 acc1 = (float8)(0.0f);  // rows 8-15
    float8 acc2 = (float8)(0.0f);  // rows 16-23
    float8 acc3 = (float8)(0.0f);  // rows 24-31

    // Linear thread ID within WG for cooperative SLM loading
    const int local_id = sg_id * SG_SIZE + sg_lid;  // 0..63

    // Cooperative A load: 32 rows × 32 cols = 1024 half elements
    // 64 threads, each loads 1024/64 = 16 elements
    // Each thread loads one row's worth of 16 consecutive elements (half a row)
    // Thread mapping: thread i loads row (i/2), col_offset (i%2)*16
    const int a_load_row = local_id / 2;
    const int a_load_col_base = (local_id % 2) * 16;

    // Prefetch first A tile into SLM buffer 0
    {
        __global const half* a_ptr = A + (m_start + a_load_row) * K + a_load_col_base;
        __local half* slm_ptr = slm_a + a_load_row * SLM_STRIDE + a_load_col_base;
        
        // Load 16 half elements
        half8 va0 = vload8(0, a_ptr);
        half8 va1 = vload8(1, a_ptr);
        vstore8(va0, 0, slm_ptr);
        vstore8(va1, 0, slm_ptr + 8);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int buf = 0;
    const int k_iters = K / TILE_K;

    for (int ki = 0; ki < k_iters; ki++) {
        int k_offset = ki * TILE_K;
        int next_buf = 1 - buf;

        // Prefetch next A tile into next SLM buffer (if not last iteration)
        if (ki < k_iters - 1) {
            int next_k = (ki + 1) * TILE_K;
            __global const half* a_ptr = A + (m_start + a_load_row) * K + next_k + a_load_col_base;
            __local half* slm_ptr = slm_a + next_buf * TILE_M * SLM_STRIDE + a_load_row * SLM_STRIDE + a_load_col_base;
            
            half8 va0 = vload8(0, a_ptr);
            half8 va1 = vload8(1, a_ptr);
            vstore8(va0, 0, slm_ptr);
            vstore8(va1, 0, slm_ptr + 8);
        }

        // Current SLM buffer base
        __local half* a_slm_base = slm_a + buf * TILE_M * SLM_STRIDE;

        // Process TILE_K=32 in two steps of 16 (matching DPAS k16)
        #pragma unroll
        for (int kk = 0; kk < 2; kk++) {
            int k_inner = kk * 16;

            // Load B tile: 16 k-rows × 16 cols for this subgroup
            // B is row-major: B[k][n], we need B[k_offset + k_inner + kr][n_start + col]
            // For DPAS, B needs to be packed as int8 per WI:
            // Each WI holds one column, 16 k-elements packed as 8 ints (2 fp16 per int)
            int8 b_val;
            __global const half* b_ptr = B + (k_offset + k_inner) * N + n_start + sg_lid;
            
            #pragma unroll
            for (int kr = 0; kr < 8; kr++) {
                half b0 = b_ptr[(kr * 2) * N];
                half b1 = b_ptr[(kr * 2 + 1) * N];
                // Pack two fp16 into one int (low=first k, high=second k)
                short s0 = as_short(b0);
                short s1 = as_short(b1);
                ((int*)&b_val)[kr] = (int)((uint)(ushort)s0 | ((uint)(ushort)s1 << 16));
            }

            // Load A from SLM and perform DPAS for each 8-row block
            // A for DPAS: short8 = 8 rows × 16 k-elements (each short = 1 fp16, but packed as k16)
            // Actually for intel_sub_group_f16_f16_matrix_mad_k16:
            //   a is short8: across the subgroup, represents 8 rows × 16 k-elements
            //   Each WI holds 8 shorts = 8 k-element pairs? No...
            //   Actually: short8 per WI, 16 WIs in SG → 8 rows, each row has 16 fp16 values
            //   WI[i] holds row[r]'s k-elements at positions determined by subgroup layout
            //   For Xe2 (exec_size=16): each WI holds one column of A's 8×16 block
            //   So WI sg_lid holds A[row+r][k + sg_lid] for r=0..7, packed as short8

            // Block 0: rows 0-7
            short8 a0;
            #pragma unroll
            for (int r = 0; r < 8; r++) {
                __local half* row_ptr = a_slm_base + r * SLM_STRIDE + k_inner + sg_lid;
                ((short*)&a0)[r] = as_short(*row_ptr);
            }
            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b_val, acc0);

            // Block 1: rows 8-15
            short8 a1;
            #pragma unroll
            for (int r = 0; r < 8; r++) {
                __local half* row_ptr = a_slm_base + (8 + r) * SLM_STRIDE + k_inner + sg_lid;
                ((short*)&a1)[r] = as_short(*row_ptr);
            }
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b_val, acc1);

            // Block 2: rows 16-23
            short8 a2;
            #pragma unroll
            for (int r = 0; r < 8; r++) {
                __local half* row_ptr = a_slm_base + (16 + r) * SLM_STRIDE + k_inner + sg_lid;
                ((short*)&a2)[r] = as_short(*row_ptr);
            }
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b_val, acc2);

            // Block 3: rows 24-31
            short8 a3;
            #pragma unroll
            for (int r = 0; r < 8; r++) {
                __local half* row_ptr = a_slm_base + (24 + r) * SLM_STRIDE + k_inner + sg_lid;
                ((short*)&a3)[r] = as_short(*row_ptr);
            }
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b_val, acc3);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
        buf = next_buf;
    }

    // Write results: each WI writes one column (sg_lid) across 32 rows
    // acc0..acc3 each have 8 float values = 8 rows for this WI's column
    __global half* c_ptr = C + m_start * N + n_start + sg_lid;

    #pragma unroll
    for (int r = 0; r < 8; r++) {
        c_ptr[r * N] = convert_half(((float*)&acc0)[r]);
    }
    #pragma unroll
    for (int r = 0; r < 8; r++) {
        c_ptr[(8 + r) * N] = convert_half(((float*)&acc1)[r]);
    }
    #pragma unroll
    for (int r = 0; r < 8; r++) {
        c_ptr[(16 + r) * N] = convert_half(((float*)&acc2)[r]);
    }
    #pragma unroll
    for (int r = 0; r < 8; r++) {
        c_ptr[(24 + r) * N] = convert_half(((float*)&acc3)[r]);
    }
}
```

Wait, I need to reconsider the A loading from SLM for DPAS. The DPAS `intel_sub_group_f16_f16_matrix_mad_k16` with exec_size=16 on Xe2:
- `a` (short8): represents an 8×16 matrix. Each WI holds one column (16 WIs = 16 k-elements), and short8 = 8 rows. So WI `i` holds `A[row+0..7][k+i]` packed as short8.
- `b` (int8): represents a 16×16 matrix. Each WI holds one column, int8 = 16 k-elements packed as pairs (2 fp16 per int, 8 ints = 16 fp16).

Wait, let me reconsider B packing. For B (16k × 16n):
- int8 per WI: 8 ints × 16 WIs. Each int has 2 fp16 values (VNNI format).
- So WI `i` holds column `i` of B, with k-elements packed in pairs: int[j] = {B[2j][col_i], B[2j+1][col_i]}.

That matches what I have. And for A:
- short8 per WI: WI `i` holds A[row+0..7][k+i], i.e., one k-element per row, 8 rows.

This also matches my SLM read pattern where `sg_lid` selects the k-column.

Let me also reconsider the SLM loading. With the current approach, the barrier placement needs to ensure the prefetch completes before we use the next buffer. Let me restructure to have the barrier at the right place.

Actually, looking more carefully at the flow:
1. Load A[0] into buf 0, barrier
2. Loop iteration 0: compute from buf 0, start loading A[1] into buf 1, barrier at end
3. Loop iteration 1: compute from buf 1, start loading A[2] into buf 0, barrier at end
...

This is correct double-buffering. But the barrier needs to be AFTER the load (to ensure it's visible) and BEFORE the compute of the next iteration. Let me restructure:

```OCL
// GEMM: C[M,N] = A[M,K] x B[K,N], all row-major FP16
// Launch: GWS = (N/16, M/32), LWS = (4, 1) with subgroup_size=16
// Effective: 4 subgroups × 16 WIs = 64 WIs per WG
// Each WG computes a 32×64 output tile
// TILE_M=32, TILE_N=64, TILE_K=32

#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable

#define TILE_M 32
#define TILE_N 64
#define TILE_K 32
#define SG_SIZE 16
#define NUM_SG 4
#define SLM_STRIDE 36

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(4, 1, 1)))
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
    const int sg_id = get_sub_group_id();
    const int sg_lid = get_sub_group_local_id();

    const int m_start = wg_m * TILE_M;
    const int n_start = wg_n * TILE_N + sg_id * SG_SIZE;

    __local half slm_a[2 * TILE_M * SLM_STRIDE];

    float8 acc0 = (float8)(0.0f);
    float8 acc1 = (float8)(0.0f);
    float8 acc2 = (float8)(0.0f);
    float8 acc3 = (float8)(0.0f);

    const int local_id = sg_id * SG_SIZE + sg_lid;
    const int a_load_row = local_id / 2;
    const int a_load_col_base = (local_id % 2) * 16;

    // Load first A tile into buffer 0
    {
        __global const half* a_ptr = A + (m_start + a_load_row) * K + a_load_col_base;
        __local half* slm_ptr = slm_a + a_load_row * SLM_STRIDE + a_load_col_base;
        half8 va0 = vload8(0, a_ptr);
        half8 va1 = vload8(1, a_ptr);
        vstore8(va0, 0, slm_ptr);
        vstore8(va1, 0, slm_ptr + 8);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const int k_iters = K / TILE_K;

    for (int ki = 0; ki < k_iters; ki++) {
        int k_offset = ki * TILE_K;
        int buf = ki % 2;
        __local half* a_slm_base = slm_a + buf * TILE_M * SLM_STRIDE;

        // Compute: process current tile from SLM
        #pragma unroll
        for (int kk = 0; kk < 2; kk++) {
            int k_inner = kk * 16;

            // Load B
            int8 b_val;
            __global const half* b_ptr = B + (k_offset + k_inner) * N + n_start + sg_lid;
            #pragma unroll
            for (int kr = 0; kr < 8; kr++) {
                half b0 = b_ptr[(kr * 2) * N];
                half b1 = b_ptr[(kr * 2 + 1) * N];
                ((int*)&b_val)[kr] = (int)((uint)(ushort)as_short(b0) | ((uint)(ushort)as_short(b1) << 16));
            }

            // Load A blocks from SLM and DPAS
            short8 a0, a1, a2, a3;
            #pragma unroll
            for (int r = 0; r < 8; r++) {
                ((short*)&a0)[r] = as_short(a_slm_base[r * SLM_STRIDE + k_inner + sg_lid]);
            }
            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b_val, acc0);

            #pragma unroll
            for (int r = 0; r < 8; r++) {
                ((short*)&a1)[r] = as_short(a_slm_base[(8 + r) * SLM_STRIDE + k_inner + sg_lid]);
            }
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b_val, acc1);

            #pragma unroll
            for (int r = 0; r < 8; r++) {
                ((short*)&a2)[r] = as_short(a_slm_base[(16 + r) * SLM_STRIDE + k_inner + sg_lid]);
            }
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b_val, acc2);

            #pragma unroll
            for (int r = 0; r < 8; r++) {
                ((short*)&a3)[r] = as_short(a_slm_base[(24 + r) * SLM_STRIDE + k_inner + sg_lid]);
            }
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b_val, acc3);
        }

        // Load next A tile (double buffer)
        if (ki < k_iters - 1) {
            int next_buf = 1 - buf;
            int next_k = (ki + 1) * TILE_K;
            __global const half* a_ptr = A + (m_start + a_load_row) * K + next_k + a_load_col_base;
            __local half* slm_ptr = slm_a + next_buf * TILE_M * SLM_STRIDE + a_load_row * SLM_STRIDE + a_load_col_base;
            half8 va0 = vload8(0, a_ptr);
            half8 va1 = vload8(1, a_ptr);
            vstore8(va0, 0, slm_ptr);
            vstore8(va1, 0, slm_ptr + 8);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store results
    __global half* c_ptr = C + m_start * N + n_start + sg_lid;
    #pragma unroll
    for (int r = 0; r < 8; r++) c_ptr[r * N] = convert_half(((float*)&acc0)[r]);
    #pragma unroll
    for (int r = 0; r < 8; r++) c_ptr[(8 + r) * N] = convert_half(((float*)&acc1)[r]);
    #pragma unroll
    for (int r = 0; r < 8; r++) c_ptr[(16 + r) * N] = convert_half(((float*)&acc2)[r]);
    #pragma unroll
    for (int r = 0; r < 8; r++) c_ptr[(24 + r) * N] = convert_half(((float*)&acc3)[r]);
}
```