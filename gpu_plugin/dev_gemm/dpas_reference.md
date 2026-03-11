# Intel DPAS (Dot Product Accumulate Systolic) API Reference for GEMM

## Overview
DPAS (also known as XMX - Xe Matrix eXtensions) is Intel's hardware-accelerated matrix multiply-accumulate instruction available on Xe architecture GPUs (including Lunar Lake/LNL). It provides significant performance improvements for GEMM operations by executing matrix operations directly in hardware.

## Hardware Architecture
- **LNL Xe2 Architecture Specs:**
  - 8 Xe Cores
  - 8 EUs per Xe Core (64 total EUs)
  - 8 threads per EU
  - Subgroup size: 16 or 32
  - 256 registers per EU (256 bytes each)
  - 32KB shared local memory per Xe Core

## DPAS Instruction Format

### Intel GPU Assembly (DPAS)
```assembly
dpas.systolic_depth.repeat_count  dst  src0  src1  src2
```

- **systolic_depth**: Number of inner loop iterations (1, 2, 4, 8)
- **repeat_count**: Number of times to repeat the operation (1-8)
- **dst**: Destination accumulator (GRF - General Register File)
- **src0**: Accumulator input (GRF)
- **src1**: Matrix A operand (GRF)
- **src2**: Matrix B operand (GRF)

### DPAS Operation
Computes: `dst = src0 + src1 * src2`

Where:
- `src1`: M×K matrix (M rows, K columns)
- `src2`: K×N matrix (K rows, N columns)  
- `dst/src0`: M×N accumulator matrix

## OpenCL DPAS Support

### Method 1: Intel Subgroup Block Read + Inline Assembly
```opencl
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable

// Intel subgroup cooperative matrix extension
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void gemm_dpas(...) {
    // Use subgroup operations
    int sg_lid = get_sub_group_local_id();
    
    // Block reads for coalesced access
    float8 a_data = intel_sub_group_block_read8((__global uint*)A);
    float8 b_data = intel_sub_group_block_read8((__global uint*)B);
    
    // Accumulator
    float8 acc = 0.0f;
    
    // DPAS operation (pseudo-code - actual implementation uses intrinsics)
    acc = intel_sub_group_f16_f16_matrix_mad_k16(acc, a_data, b_data);
}
```

### Method 2: SYCL/DPC++ DPAS Intrinsics
```cpp
// SYCL joint_matrix extension (most portable)
#include <sycl/ext/oneapi/experimental/matrix/matrix.hpp>

using namespace sycl::ext::oneapi::experimental::matrix;

// Matrix dimensions for DPAS
constexpr int M = 8;  // Rows of A and C
constexpr int N = 16; // Columns of B and C
constexpr int K = 16; // Columns of A, rows of B

// Define matrix types
using matrix_a = joint_matrix<sub_group, half, use::a, M, K, layout::row_major>;
using matrix_b = joint_matrix<sub_group, half, use::b, K, N, layout::row_major>;
using matrix_c = joint_matrix<sub_group, float, use::accumulator, M, N>;

// In kernel:
matrix_a a;
matrix_b b;
matrix_c c;

// Load matrices
joint_matrix_load(sg, a, A_ptr, stride_a);
joint_matrix_load(sg, b, B_ptr, stride_b);
joint_matrix_fill(sg, c, 0.0f);

// DPAS operation
joint_matrix_mad(sg, c, a, b, c);

// Store result
joint_matrix_store(sg, c, C_ptr, stride_c);
```

### Method 3: OpenCL C with Intel Extensions
```opencl
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable

// Use Intel matrix operations
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void gemm_dpas_optimized(
    const int M,
    const int N, 
    const int K,
    __global const half* A,    // fp16 input
    __global const half* B,    // fp16 input
    __global float* C)         // fp32 output
{
    // Work-item computes 8x16 output block
    const int m_block = 8;
    const int n_block = 16;
    const int k_block = 16;
    
    int sg_id = get_sub_group_id();
    int sg_lid = get_sub_group_local_id();
    
    // Accumulators (8x16 per work-item)
    float8 acc0 = 0.0f;
    float8 acc1 = 0.0f;
    
    // Loop over K dimension
    for (int k = 0; k < K; k += k_block) {
        // Load A block (8x16) - cooperative within subgroup
        half8 a_data = intel_sub_group_block_read_us8((__global ushort*)(A + offset_a));
        
        // Load B block (16x16) - cooperative within subgroup  
        half8 b_data0 = intel_sub_group_block_read_us8((__global ushort*)(B + offset_b0));
        half8 b_data1 = intel_sub_group_block_read_us8((__global ushort*)(B + offset_b1));
        
        // DPAS computation
        // Each subgroup computes 8x16 output using 8x16 (A) × 16x16 (B)
        acc0 = intel_sub_group_f16_f16_matrix_mad_k16(acc0, a_data, b_data0);
        acc1 = intel_sub_group_f16_f16_matrix_mad_k16(acc1, a_data, b_data1);
    }
    
    // Store results
    intel_sub_group_block_write8((__global uint*)(C + offset_c), acc0);
    intel_sub_group_block_write8((__global uint*)(C + offset_c + 8*N), acc1);
}
```

## DPAS Configuration for Optimal Performance

### Supported Data Type Combinations
| src1 (A) | src2 (B) | dst (C) | Systolic Depth | Notes |
|----------|----------|---------|----------------|-------|
| fp16     | fp16     | fp32    | 8              | Best for ML inference |
| bf16     | bf16     | fp32    | 8              | Training workloads |
| int8     | int8     | int32   | 8              | Quantized inference |
| int4     | int4     | int32   | 8              | Ultra-low precision |
| int8     | uint8    | int32   | 8              | Mixed signedness |
| tf32     | tf32     | fp32    | 4              | High precision |

### Optimal Matrix Block Sizes
- **Subgroup size 16**: 8×16 output per subgroup
- **Subgroup size 32**: 16×16 output per subgroup
- **K-dimension**: Multiples of 16 for best efficiency
- **Tile sizes**: 128×128 or 256×256 for L1 cache optimization

## GEMM Implementation with DPAS

### Key Optimization Strategy
```
1. Tile matrices into blocks fitting in local memory (32KB)
2. Each work-group: 16x16 work-items (subgroups)
3. Each subgroup (16 threads): computes 8×16 output elements
4. Load tiles cooperatively using block reads
5. Perform DPAS ops (8×16 × 16×16 = 8×16 output)
6. Accumulate in FP32 for numerical stability
7. Store results with block writes
```

### Memory Layout for DPAS
```
Matrix A (M×K): Row-major preferred
- Each subgroup loads 8 rows × 16 cols
- Use intel_sub_group_block_read for coalesced access

Matrix B (K×N): Column-major or packed layout
- Transpose or pack for efficient DPAS access
- Consider pre-transposing B offline

Matrix C (M×N): Row-major
- Each subgroup writes 8×16 block
- Use intel_sub_group_block_write for coalescing
```

### Performance Targets with DPAS
- **Theoretical Peak (LNL Xe2)**: ~2048 GFLOPS FP16
- **Realistic Target (80%)**: ~1640 GFLOPS
- **With DPAS optimization**: 70-85% efficiency achievable
- **Without DPAS**: 30-50% efficiency (as we currently have)

## Example: Complete DPAS-based GEMM Kernel

```opencl
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void gemm_dpas_fp16(
    const int M,
    const int N,
    const int K,
    __global const ushort* A,  // fp16, row-major, M×K
    __global const ushort* B,  // fp16, row-major, K×N
    __global float* C)          // fp32, row-major, M×N
{
    // Each work-group handles 128×128 tile
    // Each subgroup handles 8×16 tile
    const int tile_m = 128;
    const int tile_n = 128;
    const int tile_k = 32;
    
    __local ushort tileA[128][32]; // 8KB
    __local ushort tileB[32][128]; // 8KB
    
    int group_m = get_group_id(0);
    int group_n = get_group_id(1);
    int sg_id = get_sub_group_id();
    int sg_lid = get_sub_group_local_id();
    
    // Each subgroup computes 8×16 output block
    int m_base = group_m * tile_m + (sg_id / 8) * 8;
    int n_base = group_n * tile_n + (sg_id % 8) * 16;
    
    // Accumulators (8 rows × 16 cols per subgroup)
    float8 acc[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        acc[i] = 0.0f;
    }
    
    // Loop over K in tiles
    for (int k = 0; k < K; k += tile_k) {
        // Cooperative load of A tile (128×32)
        // ... cooperative loading logic ...
        
        // Cooperative load of B tile (32×128)
        // ... cooperative loading logic ...
        
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // DPAS computation on tiles
        #pragma unroll
        for (int kk = 0; kk < tile_k; kk += 16) {
            // Load 8×16 from A
            ushort8 a_vec[2];
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                // Load via subgroup block read
            }
            
            // Load 16×16 from B  
            ushort8 b_vec[16];
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                // Load via subgroup block read
            }
            
            // DPAS: 8×16 = (8×16) × (16×16)
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                acc[i] = intel_sub_group_f16_f16_matrix_mad_k16(
                    acc[i], a_vec, b_vec[i]);
            }
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // Store 8×16 output block
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int row = m_base + i;
        if (row < M) {
            intel_sub_group_block_write8(
                (__global uint*)&C[row * N + n_base], 
                acc[i * 2]);
            intel_sub_group_block_write8(
                (__global uint*)&C[row * N + n_base + 8],
                acc[i * 2 + 1]);
        }
    }
}
```

## Compiler Flags for DPAS
```bash
# OpenCL
clang -cl-std=CL2.0 -cl-intel-greater-than-4GB-buffer-required \
      -cl-intel-enable-auto-large-GRF-mode \
      -cl-intel-256-GRF-per-thread

# DPC++/SYCL
icpx -fsycl -fsycl-targets=spir64_gen -Xs "-device xe_hpg" \
     -DMKL_ILP64 -DMKL_DIRECT_CALL
```

## Performance Analysis
```
DPAS Instruction Throughput:
- 8 DPAS ops per cycle per EU
- 64 EUs × 8 ops × 2.0 GHz = 1024 DPAS/cycle
- Each DPAS: 8×16×16 = 2048 FP16 ops
- Peak: 1024 × 2048 × 2.0 = ~4194 GFLOPS (FP16)
- FP32 accumulate: ~2048 GFLOPS effective

Bottlenecks:
1. Memory bandwidth: 128 GB/s typical
2. Register pressure: 256 registers/thread
3. Local memory: 32KB shared
4. Synchronization overhead
```

## Debugging Tips
1. **Verify DPAS support**: Check GPU capabilities
2. **Alignment**: Ensure 16-byte alignment for block reads
3. **Subgroup operations**: Validate subgroup size is 16
4. **Numeric accuracy**: Use FP32 accumulators for stability
5. **Profiling**: Use VTune/GPA to verify DPAS utilization

## References
- [Intel Xe Architecture Whitepaper](https://www.intel.com/content/www/us/en/developer/articles/technical/xe-gpu-architecture.html)
- [oneAPI DPC++ Matrix Extensions](https://github.com/intel/llvm/tree/sycl/sycl/doc/extensions/experimental/matrix)
- [OpenCL Intel Extensions](https://registry.khronos.org/OpenCL/extensions/intel/)
- [Intel GPU ISA Documentation](https://www.intel.com/content/www/us/en/develop/documentation/intel-graphics-compiler-developer-guide/top.html)
