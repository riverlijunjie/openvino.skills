/*
[USER_INSTRUCTIONS_START]
Consider some kernel optimizations tips :
    - For compute bound kernels, focus on improving FLOPS and reducing or hiding memory access latency.
    - For memory bound kernels, focus on improving memory bandwidth utilization and access patterns.
    - Use local memory to reduce global memory access latency if possible.
    - Optimize work-group sizes based on the GPU architecture.
    - Minimize divergent branches within the kernel.
    - Unroll loops where applicable to increase instruction-level parallelism.
    - Use vectorized data types and block read to improve memory bandwidth utilization.
    - Optimize memory access patterns to ensure coalesced accesses, which can significantly improve performance.
    - Consider tiling techniques to maximize data reuse in local memory.
    - Use fused dpas(XMX) operations for compute bound kernels as much as possible, and optimize data layout and access patterns to maximize DPAS efficiency.
    - Implement double buffering to overlap computation with memory transfers.
    - For int4 and int8 data types, consider using efficient bitwise unpacking techniques to optimize data loading.
    - Leverage Intel subgroup extensions for optimized collective operations and DPAS instructions if available.
    - Avoid register spilling by carefully managing register usage and considering the use of local memory for intermediate storage when necessary.
    - Must provide GWS/LWS and sub_group_size information in the kernel code comments to guide the search process.
[USER_INSTRUCTIONS_END]
*/

// [EVOLVE_START]
kernel void gemm(__global const half* A,
                 __global const half* B,
                 __global half* C,
                 const int M,
                 const int K,
                 const int N)
{
}
// [EVOLVE_END]
