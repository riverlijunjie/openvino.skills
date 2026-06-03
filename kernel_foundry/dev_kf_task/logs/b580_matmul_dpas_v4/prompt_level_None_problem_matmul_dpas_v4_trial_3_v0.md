

# You are a OCL programming expert specializing in GPU kernel optimization. 
Given a reference OCL implementation, your objective is to create a performant OCL kernel with identical functionality as the reference.

The code you generate will be pasted into an existing project. Make sure to follow the existing code structure and function signatures.

## The user provided the following additional instructions for you:
Optimization goals:
- Must use Intel OpenCL DPAS instruction, e.g. intel_sub_group_f16_f16_matrix_mad_k16.
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

### Version 1 (Result: Correctness score: 5 / 5 (compiles and correct), runtime in ms: 30.700):
```OCL
// Tiled FP16 matmul with SLM + DPAS (XMX)
// C[M,N] = A[M,K] x B[K,N], all half, accumulate in float
//
// WG tile: 32 rows x 32 cols
// K-tile: 16
// 8 subgroups per WG in 4x2 grid (4 along M, 2 along N)
// Each subgroup: 8 rows x 16 cols via intel_sub_group_f16_f16_matrix_mad_k16
//
// Launch metadata:
//   Subgroup size: 16
//   LWS: (16, 8, 1) = 128 work-items = 8 subgroups
//   GWS: (ceil(N/32)*16, ceil(M/32)*8, 1)

#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define TILE_M 32
#define TILE_N 32
#define TILE_K 16
#define SG_ROWS 8
#define NUM_SG 8

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(16, 8, 1)))
__kernel void matmul(
    __global const half* restrict A,
    __global const half* restrict B,
    __global half* restrict C,
    const int M,
    const int K,
    const int N)
{
    // SLM tiles - padded to avoid bank conflicts
    __local half A_slm[TILE_M * (TILE_K + 2)];  // 32 x 18 = 576 halfs
    __local half B_slm[TILE_K * (TILE_N + 2)];  // 16 x 34 = 544 halfs

    const int sg_lid = get_sub_group_local_id();
    const int sg_id  = get_sub_group_id();        // 0..7

    // SG grid: sg_row = sg_id / 2 (0..3), sg_col = sg_id % 2 (0..1)
    const int sg_row = sg_id >> 1;
    const int sg_col = sg_id & 1;

    const int wg_m = get_group_id(1) * TILE_M;
    const int wg_n = get_group_id(0) * TILE_N;

    const int row_base = wg_m + sg_row * SG_ROWS;
    const int col_base = wg_n + sg_col * 16;

    // Flat local ID for cooperative loading
    const int lid = get_local_id(0) + get_local_id(1) * 16; // 0..127
    const int A_STRIDE = TILE_K + 2;
    const int B_STRIDE = TILE_N + 2;

    float8 acc = 0.0f;

    for (int k = 0; k < K; k += TILE_K) {
        // Cooperative load A_slm[32][16]: 512 elements, 128 WIs -> 4 each
        for (int i = lid; i < TILE_M * TILE_K; i += NUM_SG * 16) {
            int r = i >> 4;        // i / 16
            int c = i & 15;        // i % 16
            int grow = wg_m + r;
            int gcol = k + c;
            A_slm[r * A_STRIDE + c] = (grow < M && gcol < K) ? A[grow * K + gcol] : (half)0.0h;
        }

        // Cooperative load B_slm[16][32]: 512 elements, 128 WIs -> 4 each
        for (int i = lid; i < TILE_K * TILE_N; i += NUM_SG * 16) {
            int r = i >> 5;        // i / 32
            int c = i & 31;        // i % 32
            int grow = k + r;
            int gcol = wg_n + c;
            B_slm[r * B_STRIDE + c] = (grow < K && gcol < N) ? B[grow * N + gcol] : (half)0.0h;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Load A from SLM: 8 rows x 16 K-cols for this subgroup
        short8 a_val;
        #pragma unroll
        for (int r = 0; r < 8; r++) {
            a_val[r] = as_short(A_slm[(sg_row * SG_ROWS + r) * A_STRIDE + sg_lid]);
        }

        // Load B from SLM: 16 K-rows x 16 N-cols for this subgroup's column block
        int8 b_val;
        int col_off = sg_col * 16;
        #pragma unroll
        for (int p = 0; p < 8; p++) {
            int k0 = 2 * p;
            int k1 = 2 * p + 1;
            half bv0 = B_slm[k0 * B_STRIDE + col_off + sg_lid];
            half bv1 = B_slm[k1 * B_STRIDE + col_off + sg_lid];
            short2 packed = (short2)(as_short(bv0), as_short(bv1));
            ((int*)&b_val)[p] = as_int(packed);
        }

        // DPAS: 8x16 * 16x16 -> 8x16
        acc = intel_sub_group_f16_f16_matrix_mad_k16(a_val, b_val, acc);

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store: each WI writes column (col_base + sg_lid) for 8 rows
    #pragma unroll
    for (int r = 0; r < 8; r++) {
        int row_idx = row_base + r;
        int col_idx = col_base + sg_lid;
        if (row_idx < M && col_idx < N) {
            C[row_idx * N + col_idx] = convert_half(((float*)&acc)[r]);
        }
    }
}
```

### Version 2 (Result: Correctness score: 5 / 5 (compiles and correct), runtime in ms: 33.900):
```OCL
// Tiled FP16 matmul using Intel DPAS (XMX) instructions
// C[M,N] = A[M,K] x B[K,N], all half, accumulate in float
// Each subgroup (16 lanes) computes 8x8 output via intel_sub_group_f16_f16_matrix_mad_k16
// Workgroup: 2 subgroups along N (16 cols), 4 subgroups along M (32 rows) => LWS=(32,4), 128 WIs
// GWS = (ceil(N/16)*32, ceil(M/32)*4)
// Subgroup size: 16
// TILE_M=32, TILE_N=16, TILE_K=16

#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void matmul(
    __global const half* A,
    __global const half* B,
    __global half* C,
    const int M,
    const int K,
    const int N)
{
    // Subgroup and workgroup identification
    const int sg_id = get_sub_group_id();          // 0..7
    const int sg_lid = get_sub_group_local_id();   // 0..15 (lane within subgroup)

    // Workgroup layout: 8 subgroups in a 4x2 grid (4 along M, 2 along N)
    // Each subgroup: 8 rows x 8 cols of C
    // WG tile: 32 rows x 16 cols
    const int sg_row = sg_id / 2;  // 0..3
    const int sg_col = sg_id % 2;  // 0..1

    const int wg_id_x = get_group_id(0);  // along N
    const int wg_id_y = get_group_id(1);  // along M

    const int base_row = wg_id_y * 32 + sg_row * 8;
    const int base_col = wg_id_x * 16 + sg_col * 8;

    // Each lane in the subgroup handles one of 8 rows, producing 8 output values (cols)
    // DPAS: acc = A_block * B_block + acc
    // A_block: 8 rows x 16 cols (each lane holds half8 = 8 rows, k-elements distributed across lanes)
    // B_block: 16 rows x 8 cols (each lane holds half8)
    // Result: 8x8 tile accumulated in float8

    float8 acc = (float8)(0.0f);

    for (int k_base = 0; k_base < K; k_base += 16) {
        // Load A tile: 8 rows x 16 cols, row-major
        // Each subgroup lane loads one column (sg_lid) across 8 rows
        half8 a_tile;
        for (int r = 0; r < 8; r++) {
            int row = base_row + r;
            int col_k = k_base + sg_lid;
            half val = (row < M && col_k < K) ? A[row * K + col_k] : (half)0.0h;
            ((half*)&a_tile)[r] = val;
        }

        // Load B tile: 16 rows x 8 cols
        // For DPAS, B is packed: each lane (sg_lid = k index 0..15) holds 8 col values
        half8 b_tile;
        {
            int k_row = k_base + sg_lid;
            for (int c = 0; c < 8; c++) {
                int col = base_col + c;
                half val = (k_row < K && col < N) ? B[k_row * N + col] : (half)0.0h;
                ((half*)&b_tile)[c] = val;
            }
        }

        // DPAS: 8x16 * 16x8 -> 8x8, accumulated in float
        acc = intel_sub_group_f16_f16_matrix_mad_k16(a_tile, b_tile, acc);
    }

    // Store 8x8 result tile
    // After DPAS, lane sg_lid holds column sg_lid%8... 
    // Actually, for intel_sub_group_f16_f16_matrix_mad_k16:
    // Each lane holds one column of the 8x8 result (8 row values)
    // Lane i holds column i (for i < 8), lanes 8-15 are duplicates or the mapping
    // The standard mapping: lane sg_lid holds results for column sg_lid (mod 8? or direct?)

    // For Xe2 DPAS with subgroup=16, the result layout:
    // Lanes 0-7 get the 8x8 result, lane i gets float8 for column i
    // Lanes 8-15 may mirror or be unused for 8-wide output

    // Safe approach: only lanes 0-7 write
    if (sg_lid < 8) {
        int col = base_col + sg_lid;
        for (int r = 0; r < 8; r++) {
            int row = base_row + r;
            if (row < M && col < N) {
                C[row * N + col] = convert_half(((float*)&acc)[r]);
            }
        }
    }
}
```

### Previous kernel and evaluation:
Here is the last OCL kernel we tested (Result: Correctness score: 5 / 5 (compiles and correct), runtime in ms: 11.600):
```OCL
// Optimized FP16 matmul using Intel XMX DPAS instructions
// C[M,N] = A[M,K] x B[K,N], all half, accumulate in float
//
// Tile: 8 rows x 16 cols per subgroup via intel_sub_group_f16_f16_matrix_mad_k16
// K processed in chunks of 16
//
// Launch metadata:
//   Subgroup size: 16
//   LWS: (16, 1, 1) — 1 subgroup per workgroup (simple mapping)
//   GWS: (ceil(N/16)*16, ceil(M/8), 1)
//   Each subgroup computes one 8x16 output tile
//
// For better occupancy, can use LWS=(16*SG_COUNT, 1, 1) and map multiple N-tiles per WG.

#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void matmul(
    __global const half* A,
    __global const half* B,
    __global half* C,
    const int M,
    const int K,
    const int N)
{
    // Each subgroup handles an 8x16 output tile
    const int sg_id = get_sub_group_id();
    const int sg_lid = get_sub_group_local_id(); // 0..15

    // Tile coordinates
    const int n_tile = (get_group_id(0) * get_num_sub_groups() + sg_id);
    const int m_tile = get_group_id(1);

    const int col_base = n_tile * 16;
    const int row_base = m_tile * 8;

    if (row_base >= M || col_base >= N)
        return;

    // Accumulator: 8 floats per work-item = 8 rows x 16 cols
    float8 acc = 0.0f;

    // Loop over K in steps of 16
    for (int k = 0; k < K; k += 16) {
        // Load A tile: 8 rows x 16 cols
        // Each WI loads one column across 8 rows -> half8
        // But for DPAS, A is distributed as: each WI holds elements for the systolic feed
        // For intel_sub_group_f16_f16_matrix_mad_k16:
        //   a: short8 per WI (8 rows, each row has k16 distributed across 16 WIs)
        //   b: int8 per WI (16x16 tile, packed)

        // Load A: 8 rows x 16 K-elements
        // Sub-group block read: each WI gets one K-element per row
        // WI sg_lid reads column sg_lid from each of 8 rows
        short8 a_val;
        for (int r = 0; r < 8; r++) {
            int row_idx = row_base + r;
            int a_idx = row_idx * K + k + sg_lid;
            half a_elem = (row_idx < M && (k + sg_lid) < K) ? A[a_idx] : (half)0.0h;
            ((short*)&a_val)[r] = as_short(a_elem);
        }

        // Load B: 16 K-rows x 16 N-cols
        // For DPAS b input: int8 per WI
        // B is 16x16 tile. Each WI (sg_lid = col within tile) reads 16 elements (K rows)
        // Packed as pairs: int = two halfs, so int8 = 16 halfs
        int8 b_val;
        for (int p = 0; p < 8; p++) {
            int k_row0 = k + 2 * p;
            int k_row1 = k + 2 * p + 1;
            int col_idx = col_base + sg_lid;

            half b0 = (k_row0 < K && col_idx < N) ? B[k_row0 * N + col_idx] : (half)0.0h;
            half b1 = (k_row1 < K && col_idx < N) ? B[k_row1 * N + col_idx] : (half)0.0h;

            short2 packed = (short2)(as_short(b0), as_short(b1));
            ((int*)&b_val)[p] = as_int(packed);
        }

        // DPAS: 8x16 = 8x16 * 16x16
        acc = intel_sub_group_f16_f16_matrix_mad_k16(a_val, b_val, acc);
    }

    // Store result: each WI writes column sg_lid for 8 rows
    for (int r = 0; r < 8; r++) {
        int row_idx = row_base + r;
        int col_idx = col_base + sg_lid;
        if (row_idx < M && col_idx < N) {
            C[row_idx * N + col_idx] = convert_half(((float*)&acc)[r]);
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
================== 4 passed, 1 deselected, 1 warning in 0.76s ==================
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

- **Shared Local Memory (SLM) Tiling**: Cache frequently accessed data in SLM using `group_local_memory_for_overwrite`. Synchronize with `group_barrier()` after writes and before reads. Use 16×16 or 32×32 tiles for float data.
- **Blocked/Tiled Algorithms**: Process input in blocks to bound peak memory. Trade recomputation for memory savings (e.g., Flash-Attention style). Maintain running accumulators across blocks with proper rescaling.
- **Hierarchical Parallelism**: Structure work at three levels - work-groups (tile assignment), sub-groups (sub-tile processing), work-items (register tile). Use sub-group shuffles to share data within sub-group without SLM.