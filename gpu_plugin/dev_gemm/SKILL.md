---
name: dev_gemm
description: Develop General Matrix-Matrix multiplication (GEMM) operations for better performance. Use when working on linear algebra computations or improving matrix-matrix multiplication efficiency.
---

When develop gemm opencl code, always include:
1. **Read OpenCL gemm kernel**: If possible, read the existing OpenCL kernel for gemm operations to understand its structure and optimizations.
2. **Read GPU specified architecture details**: Search for specific GPU architecture details to tailor optimizations effectively.
3. **Understand the gemm operation**: 
    - gemm stands for General Matrix-Matrix multiplication, which is a fundamental operation in linear algebra.
    - It involves multiplying a matrix (A) by a matrix (B) to produce another matrix (C). The operation can be expressed as C = A * B + D, where D is an optional bias matrix.
    - The performance of gemm can be significantly affected by factors such as memory access patterns, data types, and the underlying hardware architecture.
4. **Identify GPU architecture**: 
    - GPU architectures: Intel GPU LNL xe2 architecture.
    - Xe Core Number: 8
    - EU number of each Xe Core: 8
    - Threads number of each EU: 8
    - Subgroup size: 16 or 32
    - Register number of each EU: 256
    - Register size: 256 bytes
    - Shared Local memory size: 32KB
5. **Get kernel optimization requirements**: 
    - Determine kernel optimization target: 80% of hardware roofline of computation.
    - Determine the Matrix A types: f16
    - Determine the Matrix A layout: col-major.
    - Determine the Matrix B types: f16 or int4 or int8
        - f16: 16-bit floating point
        - int4: 4-bit integer, group quantization format or channel quantization format 
        - int8: 8-bit integer, group quantization format or channel quantization format
    - Determine the Matrix B layout: col-major.
    - Determine the matrix dimensions: unlimited height and width and aligned to 16.
    - Determine the Matrix D types: f32
4. **Consider some kernel optimizations tips**:
    - gemm is compute bound, focus on improving FLOPS and reducing or hiding memory access latency.
    - Use local memory to reduce global memory access latency if possible.
    - Optimize work-group sizes based on the GPU architecture.
    - Minimize divergent branches within the kernel.
    - Unroll loops where applicable to increase instruction-level parallelism.
    - Use vectorized data types and block read to improve memory bandwidth utilization.
    - Optimize memory access patterns to ensure coalesced accesses, which can significantly improve performance.
    - Consider tiling techniques to maximize data reuse in local memory.
    - Consider using fused dpas(XMX) operations if supported by the hardware.
    - Implement double buffering to overlap computation with memory transfers.
    - For int4 and int8 data types, consider using efficient bitwise unpacking techniques to optimize data loading.
    - Leverage Intel subgroup extensions for optimized collective operations and DPAS instructions if available.
    - Avoid register spilling by carefully managing register usage and considering the use of local memory for intermediate storage when necessary.
5. **Implement the optimized gemm kernel**: 
    - Write the OpenCL code for the optimized gemm operation based on the gathered requirements and optimization tips.
6. **Write test case**: 
    - Create separated test cases to validate the correctness and performance of the optimized gemm kernel.
    - Test kernel performance with different matrix sizes and data types.
    - Test kernel performance against the hardware roofline and print test results including the ratio to hardware roofline.
7. **Document optimization**:
    - Summarize the optimizations applied and their impact on performance.
    - Include before-and-after performance metrics to demonstrate improvements.
    - Include the ratio to hardware roofline to show efficiency gains.

Keep explanations conversational. For complex concepts, use multiple analogies.