

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

## Best implementation so far:

This is the code of the best kernel we have tested so far (Result: Correctness score: 5 / 5 (compiles and correct), runtime in ms: 11.600):
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

### Previous kernel and evaluation:
Here is the last OCL kernel we tested (Result: Correctness score: 3 / 5 (compiled but incorrect), runtime in ms: -1.000):
```OCL
// Optimized FP16 matmul with SLM tiling for Intel Battlemage XMX
// C[M,N] = A[M,K] x B[K,N], all half, accumulate in float
//
// WG tile: 32 rows x 16 cols (4 subgroups x 8 rows each, all share 16 cols)
// Each subgroup: 8x16 output via intel_sub_group_f16_f16_matrix_mad_k16
// K tile: 16, B tile cached in SLM (16x16 halfs = 512 bytes)
// A tile cached in SLM (32x16 halfs = 1024 bytes), total SLM = 1536 bytes
//
// Launch metadata:
//   Subgroup size: 16
//   LWS: (64, 1, 1) — 4 subgroups per workgroup
//   GWS: (ceil(N/16)*64, ceil(M/32), 1)

#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define WG_TILE_M 32
#define WG_TILE_N 16
#define SG_TILE_M 8
#define TILE_K 16
#define NUM_SG 4

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void matmul(
    __global const half* restrict A,
    __global const half* restrict B,
    __global half* restrict C,
    const int M,
    const int K,
    const int N)
{
    // SLM for B tile (16 K-rows x 16 cols) and A tile (32 rows x 16 K-cols)
    __local half slm_B[16 * 16];   // 512 bytes
    __local half slm_A[32 * 16];   // 1024 bytes

    const int sg_id = get_sub_group_id();       // 0..3
    const int sg_lid = get_sub_group_local_id(); // 0..15

    const int n_tile = get_group_id(0);
    const int m_tile = get_group_id(1);

    const int col_base = n_tile * WG_TILE_N;
    const int row_base = m_tile * WG_TILE_M + sg_id * SG_TILE_M;

    float8 acc = 0.0f;

    for (int k = 0; k < K; k += TILE_K) {
        // Cooperative load of B tile into SLM: 16x16 = 256 halfs, 64 WIs -> 4 each
        {
            int lid = sg_id * 16 + sg_lid; // 0..63
            // Each WI loads 4 elements
            for (int i = lid; i < 256; i += 64) {
                int kr = i / 16;  // k-row 0..15
                int nc = i % 16;  // n-col 0..15
                int gk = k + kr;
                int gn = col_base + nc;
                slm_B[kr * 16 + nc] = (gk < K && gn < N) ? B[gk * N + gn] : (half)0.0h;
            }
        }

        // Cooperative load of A tile into SLM: 32x16 = 512 halfs, 64 WIs -> 8 each
        {
            int lid = sg_id * 16 + sg_lid;
            for (int i = lid; i < 512; i += 64) {
                int mr = i / 16;  // m-row 0..31
                int kc = i % 16;  // k-col 0..15
                int gm = m_tile * WG_TILE_M + mr;
                int gk = k + kc;
                slm_A[mr * 16 + kc] = (gm < M && gk < K) ? A[gm * K + gk] : (half)0.0h;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Load A from SLM: 8 rows x 16 cols for this subgroup
        short8 a_val;
        int a_slm_row_base = sg_id * SG_TILE_M;
        for (int r = 0; r < 8; r++) {
            ((short*)&a_val)[r] = as_short(slm_A[(a_slm_row_base + r) * 16 + sg_lid]);
        }

        // Load B from SLM: 16x16, packed as int8
        int8 b_val;
        for (int p = 0; p < 8; p++) {
            short s0 = as_short(slm_B[(2 * p) * 16 + sg_lid]);
            short s1 = as_short(slm_B[(2 * p + 1) * 16 + sg_lid]);
            short2 packed = (short2)(s0, s1);
            ((int*)&b_val)[p] = as_int(packed);
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        acc = intel_sub_group_f16_f16_matrix_mad_k16(a_val, b_val, acc);
    }

    // Store 8x16 result
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

task.py::TestMatmulOCL::test_correctness_wrt_pytorch[0] FAILED           [ 25%]
task.py::TestMatmulOCL::test_correctness_wrt_pytorch[1] FAILED           [ 50%]
task.py::TestMatmulOCL::test_correctness_wrt_reference[0] FAILED         [ 75%]
task.py::TestMatmulOCL::test_correctness_wrt_reference[1] FAILED         [100%]

=================================== FAILURES ===================================
________________ TestMatmulOCL.test_correctness_wrt_pytorch[0] _________________

self = <task.TestMatmulOCL object at 0x72c45564f5f0>
kernel = <function initialize_matmul_kernel.<locals>.run at 0x72c45569f060>
ocl_queue = <pyopencl._cl.CommandQueue object at 0x72c4556a58a0>, _run = 0

    @pytest.mark.parametrize("_run", range(2))
    def test_correctness_wrt_pytorch(self, kernel, ocl_queue, _run):
        args, expected = get_data_on_device(DEFAULT_SHAPE, ocl_queue, seed=_run)
        kernel(*args)

        _, _, out_buf, m, _, n = args
        got_flat = np.empty(int(m) * int(n), dtype=np.float16)
        cl.enqueue_copy(ocl_queue, got_flat, out_buf)
        got = got_flat.reshape((int(m), int(n))).astype(np.float32)

        assert got.shape == expected.shape
>       assert np.allclose(got, expected, rtol=2e-2, atol=2e-2), "matmul result mismatch vs pytorch/numpy"
E       AssertionError: matmul result mismatch vs pytorch/numpy
E       assert False
E        +  where False = <function allclose at 0x72c4add4bef0>(array([[-12.4375   , -85.25     ,  45.875    , ...,   0.       ,\n          0.       ,   0.       ],\n       [-82.9375   ,  28.328125 ,   4.3085938, ...,   0.       ,\n          0.       ,   0.       ],\n       [ 31.09375  , -51.78125  ,  -9.3046875, ...,   0.       ,\n          0.       ,   0.       ],\n       ...,\n       [-65.3125   , -27.734375 ,  74.1875   , ...,   0.       ,\n          0.       ,   0.       ],\n       [ 44.6875   ,   2.3144531,  22.609375 , ...,   0.       ,\n          0.       ,   0.       ],\n       [ 50.40625  ,  -3.015625 ,  21.546875 , ...,   0.       ,\n          0.       ,   0.       ]], shape=(2048, 2048), dtype=float32), array([[-12.434087 , -85.22102  ,  45.86866  , ..., -67.074715 ,\n        -64.52674  ,  37.798523 ],\n       [-82.95244  ,  28.332115 ,   4.3084497, ...,  37.17192  ,\n         48.87541  ,  55.1519   ],\n       [ 31.096529 , -51.77693  ,  -9.3054905, ...,   8.124319 ,\n         61.21928  ,   4.7092314],\n       ...,\n       [-65.29967  , -27.73106  ,  74.195465 , ..., 122.09403  ,\n        -41.569603 ,  10.711429 ],\n       [ 44.6838   ,   2.3142765,  22.61605  , ..., -35.807106 ,\n         42.793472 ,  52.60636  ],\n       [ 50.399834 ,  -3.015791 ,  21.545517 , ..., -21.399685 ,\n        -36.035267 ,  49.01544  ]], shape=(2048, 2048), dtype=float32), rtol=0.02, atol=0.02)
E        +    where <function allclose at 0x72c4add4bef0> = np.allclose

task.py:99: AssertionError
________________ TestMatmulOCL.test_correctness_wrt_pytorch[1] _________________

self = <task.TestMatmulOCL object at 0x72c4ad72e330>
kernel = <function initialize_matmul_kernel.<locals>.run at 0x72c45569f060>
ocl_queue = <pyopencl._cl.CommandQueue object at 0x72c4556a58a0>, _run = 1

    @pytest.mark.parametrize("_run", range(2))
    def test_correctness_wrt_pytorch(self, kernel, ocl_queue, _run):
        args, expected = get_data_on_device(DEFAULT_SHAPE, ocl_queue, seed=_run)
        kernel(*args)

        _, _, out_buf, m, _, n = args
        got_flat = np.empty(int(m) * int(n), dtype=np.float16)
        cl.enqueue_copy(ocl_queue, got_flat, out_buf)
        got = got_flat.reshape((int(m), int(n))).astype(np.float32)

        assert got.shape == expected.shape
>       assert np.allclose(got, expected, rtol=2e-2, atol=2e-2), "matmul result mismatch vs pytorch/numpy"
E       AssertionError: matmul result mismatch vs pytorch/numpy
E       assert False
E        +  where False = <function allclose at 0x72c4add4bef0>(array([[ 63.21875  ,  13.6640625,  62.71875  , ...,   0.       ,\n          0.       ,   0.       ],\n       [ 30.34375  ,  -9.578125 , -15.8515625, ...,   0.       ,\n          0.       ,   0.       ],\n       [ -9.828125 ,  26.84375  , -39.875    , ...,   0.       ,\n          0.       ,   0.       ],\n       ...,\n       [-50.9375   , -10.71875  , -18.65625  , ...,   0.       ,\n          0.       ,   0.       ],\n       [-41.46875  ,  -5.0351562,  35.5      , ...,   0.       ,\n          0.       ,   0.       ],\n       [-33.90625  ,  51.46875  ,  13.109375 , ...,   0.       ,\n          0.       ,   0.       ]], shape=(2048, 2048), dtype=float32), array([[  63.220627 ,   13.663691 ,   62.708282 , ...,   26.950535 ,\n        -100.14888  ,  -76.10468  ],\n       [  30.338015 ,   -9.576593 ,  -15.848044 , ...,  -86.66203  ,\n           6.3691177,    9.569207 ],\n       [  -9.825886 ,   26.83852  ,  -39.88768  , ...,   94.32298  ,\n         -40.437588 ,   13.349518 ],\n       ...,\n       [ -50.946926 ,  -10.7210655,  -18.652342 , ...,   -4.0612535,\n         -29.112085 ,   -2.7683525],\n       [ -41.46417  ,   -5.034666 ,   35.500336 , ...,    3.5289268,\n          14.26104  ,   55.58531  ],\n       [ -33.896618 ,   51.45737  ,   13.108513 , ...,   11.92079  ,\n         -64.022385 ,   63.048595 ]], shape=(2048, 2048), dtype=float32), rtol=0.02, atol=0.02)
E        +    where <function allclose at 0x72c4add4bef0> = np.allclose

task.py:99: AssertionError
_______________ TestMatmulOCL.test_correctness_wrt_reference[0] ________________

self = <task.TestMatmulOCL object at 0x72c455692b10>
kernel = <function initialize_matmul_kernel.<locals>.run at 0x72c45569f060>
ocl_queue = <pyopencl._cl.CommandQueue object at 0x72c4556a58a0>, _run = 0

    @pytest.mark.parametrize("_run", range(2))
    def test_correctness_wrt_reference(self, kernel, ocl_queue, _run):
        args, _ = get_data_on_device(DEFAULT_SHAPE, ocl_queue, seed=100 + _run)
        kernel(*args)

        _, _, out_buf, m, _, n = args
        got_flat = np.empty(int(m) * int(n), dtype=np.float16)
        cl.enqueue_copy(ocl_queue, got_flat, out_buf)
        got = got_flat.reshape((int(m), int(n))).astype(np.float32)

        ref_kernel = initialize_matmul_kernel("matmul_reference.cl", ocl_queue)
        ref_kernel(*args)
        ref_flat = np.empty(int(m) * int(n), dtype=np.float16)
        cl.enqueue_copy(ocl_queue, ref_flat, out_buf)
        ref = ref_flat.reshape((int(m), int(n))).astype(np.float32)

>       assert np.allclose(got, ref, rtol=2e-2, atol=2e-2), "matmul result mismatch vs reference"
E       AssertionError: matmul result mismatch vs reference
E       assert False
E        +  where False = <function allclose at 0x72c4add4bef0>(array([[ -57.125    ,  -28.015625 ,   65.5625   , ...,    0.       ,\n           0.       ,    0.       ],\n       [  92.6875   ,   18.34375  ,  -14.34375  , ...,    0.       ,\n           0.       ,    0.       ],\n       [  -2.5585938,  -30.015625 ,   49.9375   , ...,    0.       ,\n           0.       ,    0.       ],\n       ...,\n       [ -46.03125  , -142.625    ,  -61.5      , ...,    0.       ,\n           0.       ,    0.       ],\n       [  31.859375 ,  -23.484375 ,   42.75     , ...,    0.       ,\n           0.       ,    0.       ],\n       [ -38.96875  ,   91.75     ,   23.953125 , ...,    0.       ,\n           0.       ,    0.       ]], shape=(2048, 2048), dtype=float32), array([[-5.7125000e+01, -2.8015625e+01,  6.5562500e+01, ...,\n        -5.0218750e+01,  2.7000000e+01,  2.6109375e+01],\n       [ 9.2687500e+01,  1.8343750e+01, -1.4343750e+01, ...,\n        -4.0843750e+01, -2.5828125e+01, -3.1914062e+00],\n       [-2.5585938e+00, -3.0015625e+01,  4.9937500e+01, ...,\n        -5.7156250e+01, -9.2250000e+01,  1.2921875e+01],\n       ...,\n       [-4.6031250e+01, -1.4262500e+02, -6.1500000e+01, ...,\n        -1.2194824e-01,  6.6445312e+00, -5.3156250e+01],\n       [ 3.1859375e+01, -2.3484375e+01,  4.2750000e+01, ...,\n         1.8390625e+01, -2.0507812e+00, -1.6000000e+02],\n       [-3.8968750e+01,  9.1750000e+01,  2.3953125e+01, ...,\n         4.5437500e+01,  5.4437500e+01,  1.0787500e+02]],\n      shape=(2048, 2048), dtype=float32), rtol=0.02, atol=0.02)
E        +    where <function allclose at 0x72c4add4bef0> = np.allclose

task.py:117: AssertionError
_______________ TestMatmulOCL.test_correctness_wrt_reference[1] ________________

self = <task.TestMatmulOCL object at 0x72c455692e10>
kernel = <function initialize_matmul_kernel.<locals>.run at 0x72c45569f060>
ocl_queue = <pyopencl._cl.CommandQueue object at 0x72c4556a58a0>, _run = 1

    @pytest.mark.parametrize("_run", range(2))
    def test_correctness_wrt_reference(self, kernel, ocl_queue, _run):
        args, _ = get_data_on_device(DEFAULT_SHAPE, ocl_queue, seed=100 + _run)
        kernel(*args)

        _, _, out_buf, m, _, n = args
        got_flat = np.empty(int(m) * int(n), dtype=np.float16)
        cl.enqueue_copy(ocl_queue, got_flat, out_buf)
        got = got_flat.reshape((int(m), int(n))).astype(np.float32)

        ref_kernel = initialize_matmul_kernel("matmul_reference.cl", ocl_queue)
        ref_kernel(*args)
        ref_flat = np.empty(int(m) * int(n), dtype=np.float16)
        cl.enqueue_copy(ocl_queue, ref_flat, out_buf)
        ref = ref_flat.reshape((int(m), int(n))).astype(np.float32)

>       assert np.allclose(got, ref, rtol=2e-2, atol=2e-2), "matmul result mismatch vs reference"
E       AssertionError: matmul result mismatch vs reference
E       assert False
E        +  where False = <function allclose at 0x72c4add4bef0>(array([[ 46.4375   , -21.75     , -36.71875  , ...,   0.       ,\n          0.       ,   0.       ],\n       [-13.859375 , -17.453125 ,  17.375    , ...,   0.       ,\n          0.       ,   0.       ],\n       [ 34.1875   ,  -9.421875 , -49.       , ...,   0.       ,\n          0.       ,   0.       ],\n       ...,\n       [  3.3261719, -60.03125  , -64.       , ...,   0.       ,\n          0.       ,   0.       ],\n       [ -0.9145508,  97.25     , -18.109375 , ...,   0.       ,\n          0.       ,   0.       ],\n       [-36.5      , -63.8125   ,  68.125    , ...,   0.       ,\n          0.       ,   0.       ]], shape=(2048, 2048), dtype=float32), array([[  46.4375   ,  -21.75     ,  -36.71875  , ...,  -39.96875  ,\n           3.125    , -147.25     ],\n       [ -13.859375 ,  -17.453125 ,   17.375    , ...,   57.90625  ,\n          25.859375 ,  -68.0625   ],\n       [  34.1875   ,   -9.421875 ,  -49.       , ...,  -46.71875  ,\n         -64.5625   ,   37.6875   ],\n       ...,\n       [   3.3261719,  -60.03125  ,  -64.       , ...,   18.328125 ,\n         -17.234375 ,  -30.90625  ],\n       [  -0.9145508,   97.25     ,  -18.109375 , ...,   35.75     ,\n         -45.0625   ,   21.78125  ],\n       [ -36.5      ,  -63.8125   ,   68.125    , ...,  -48.09375  ,\n         -29.515625 ,   11.1640625]], shape=(2048, 2048), dtype=float32), rtol=0.02, atol=0.02)
E        +    where <function allclose at 0x72c4add4bef0> = np.allclose

task.py:117: AssertionError
=============================== warnings summary ===============================
task.py::TestMatmulOCL::test_correctness_wrt_pytorch[0]
  /home/openvino-ci-74/miniforge3/envs/kernel_intel/lib/python3.12/site-packages/pyopencl/cache.py:517: CompilerWarning: Non-empty compiler output encountered. Set the environment variable PYOPENCL_COMPILER_OUTPUT=1 to see more.
    _create_built_program_from_source_cached(

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ============================
FAILED task.py::TestMatmulOCL::test_correctness_wrt_pytorch[0] - AssertionErr...
FAILED task.py::TestMatmulOCL::test_correctness_wrt_pytorch[1] - AssertionErr...
FAILED task.py::TestMatmulOCL::test_correctness_wrt_reference[0] - AssertionE...
FAILED task.py::TestMatmulOCL::test_correctness_wrt_reference[1] - AssertionE...
================== 4 failed, 1 deselected, 1 warning in 0.89s ==================

## Hardware specification:
Your code will run on the following hardware:
**Intel Battlemage** with specs: Xe-cores: 20, Render Slices: 5, Ray Tracing Units: 20, Intel® Xe Matrix Extensions (Intel® XMX) Engines: 160, Xe Vector Engines: 160, Graphics Clock: 2670, GPU Peak TOPS (Int8): 233, TBP: 190, PCI Express Configurations ‡: PCI Express 4.0 x8, Device ID: 0xE20B, Memory: 12 GB GDDR6, Memory Interface: 192 bit, Memory Bandwidth: 456, Memory Speed: 19, ISA_GPU: Xe2-HPG
Please consider the hardware specifications when improving the code. 

## Task:

**Your objectives**:
1. Analyze the previous kernel and its evaluation log.
2. Identify any errors or mismatches with the reference implementation.
3. Propose specific improvements or fixes, explaining your reasoning step by step.
4. Rewrite the kernel, providing the complete, corrected code in a code block.

**Critical Requirements:**

1. The kernel must exactly match the reference implementation's functionality.
2. The code must compile and run properly on the GPU.
3. Do not cache or reuse previous results; ensure the code executes fully on each run.
4. Keep all hyperparameters (e.g., batch size, dimensions) unchanged as specified in the reference implementation.
8. Beware of the critical error "Unexpected kernel lambda size. In such cases removing constexpr specifier aligns the captures between the host compiler and the device compiler"! Do not capture constexpr variables in lambda functions passed to kernel launches as this can lead to different lambda sizes between the host and device compiler.

Additional Guidance:

1. Clearly comment on any fixes and optimizations you implement.
2. If you change the kernel structure, explain why.
3. Anticipate and address possible runtime or compilation errors.

Please structure your response as follows:

1. Analysis:
    * Summarize the errors found in the previous kernel and log.
    * Explain your proposed changes.
2. Improved OCL code:
    * Provide the complete, corrected OCL code in a code block:
```OCL
Your code here
```


## Required Optimizations

Apply the following optimization techniques in your implementation:

- **Kernel Fusion**: Combine sequential operations (e.g., exp → add → activation) into a single kernel. Eliminate intermediate buffers by computing in registers.
- **Sub-Group Collectives**: Use `reduce_over_group(sg, val, op)` for hardware-accelerated SIMD reductions. Use `group_broadcast` and `shift_group_*` for efficient data sharing.