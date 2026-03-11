---
name: dev_gemv
description: Develop General Matrix-Vector multiplication (GEMV) operations for better performance. Use when working on linear algebra computations or improving matrix-vector multiplication efficiency.
---

When develop gemv opencl code, always include:

1. **Read OpenCL gemv kernel**: If possible, read the existing OpenCL kernel for GEMV operations to understand its structure and optimizations.
2. **Read GPU specified architecture details**: Search for specific GPU architecture details to tailor optimizations effectively.
3. **Understand the GEMV operation**: 
    - GEMV stands for General Matrix-Vector multiplication, which is a fundamental operation in linear algebra.
    - It involves multiplying a matrix (A) by a vector (x) to produce another vector (y). The operation can be expressed as y = A * x + b, where b is an optional bias vector.
    - The performance of GEMV can be significantly affected by factors such as memory access patterns, data types, and the underlying hardware architecture.
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
    - Determine the input data types used in the GEMV operation: f16
    - Determine the matrix data types used in the GEMV operation: f16 or int4 or int8
        - f16: 16-bit floating point
        - int4: 4-bit integer, group quantization format or channel quantization format 
        - int8: 8-bit integer, group quantization format or channel quantization format
    - Identify the memory layout of the matrix: col-major.
    - Understand the matrix dimensions: unlimited height and width and aligned to 16.
4. **Consider some kernel optimizations tips**:
    - gemv is memroy bound, focus on reducing memory access latency.
    - Use local memory to reduce global memory access latency if possible.
    - Optimize work-group sizes based on the GPU architecture.
    - Minimize divergent branches within the kernel.
    - Unroll loops where applicable to increase instruction-level parallelism.
    - Use vectorized data types and block read to improve memory bandwidth utilization.
    - One subgroup can process one block of 16 or 32 elements, which can be used to optimize memory access patterns.
    - Consider using fused multiply-add (FMA) operations if supported by the hardware.
    - Implement double buffering to overlap computation with memory transfers.
    - For int4 and int8 data types, consider using efficient bitwise unpacking techniques to optimize data loading.
5. **Implement the optimized GEMV kernel**: 
    - Write the OpenCL code for the optimized GEMV operation based on the gathered requirements and optimization tips.
6. **Write test case**: 
    - Create separated test cases to validate the correctness and performance of the optimized GEMV kernel.
    - Test kernel performance with different matrix sizes and data types.
    - Test kernel performance against the hardware roofline and print test results including the ratio to hardware roofline.
7. **Document optimization**:
    - Summarize the optimizations applied and their impact on performance.
    - Include before-and-after performance metrics to demonstrate improvements.
    - Include the ratio to hardware roofline to show efficiency gains.

Keep explanations conversational. For complex concepts, use multiple analogies.