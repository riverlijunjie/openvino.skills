---
name: dev_gemv_opt
description: Develop General Matrix-Vector multiplication (GEMV) operations for better performance. Use when working on linear algebra computations or improving matrix-vector multiplication efficiency.
---

When develop gemv opencl code, always include:

1. **First top priority needed**:
    - Don't create new commit
    - Don't push code to remote repository
    - Don't submit PR
    - Don't change weights layout
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
        - Video Memory Bandwidth: 100 GB/s
    - GPU architectures: Intel GPU B580(BMG) architecture.
        - Xe Core Number: 20
        - EU number of each Xe Core: 8
        - Threads number of each EU: 8
        - Subgroup size: 16 or 32
        - Register number of each EU: 256
        - Register size: 256 bytes
        - Shared Local memory size: 32KB
        - Video Memory Bandwidth: 456 GB/s
5. **Get kernel optimization requirements**: 
    - Determine the input data types used in the GEMV operation: f16 or int8
    - Determine the matrix data types used in the GEMV operation: int4 or int8
        - int4: 4-bit integer, group quantization format or channel quantization format 
        - int8: 8-bit integer, group quantization format or channel quantization format
    - Identify the memory layout of the matrix: col-major.
    - Understand the matrix dimensions: unlimited height and width and aligned to 16.
    - Need optimization:
            input tensor: f16:bfyx:1x1x4096:nopad
            weight tensor: u4:bfyx:6144x4096:nopad
            weight scale tensor: f16:fbyx:6144x32:nopad
            weight zp tensor: u8:fbyx:6144x32:nopad
            bias: f16:bfyx:0x1x1:nopad
            output tensor: f16:bfyx:1x1x6144:nopad
5. **Consider some kernel optimizations tips**:
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
6. **Implement the optimized GEMV kernel**: 
    - Write the OpenCL code for the optimized GEMV operation based on the gathered requirements and optimization tips.
7. **Write test case**: 
    - Create separated test cases to validate the correctness and performance of the optimized GEMV kernel.
    - Test kernel performance with different matrix sizes and data types.
    - Test kernel performance against the hardware roofline and print test results including the ratio to hardware roofline.
8. **Remote build and test**:
    - Remote copy modified code to remote machine:
        - Remote machine: openvino-ci-74@10.239.140.155, pwd: openvino
        - Remote openvino directory: /mnt/river/moe/openvino
        - Use secure copy (scp) or a similar method to transfer the modified code to the remote machine's openvino directory
    - Set up a remote build and test environment to validate the optimized GEMV kernel on the target GPU architecture.
    - Remote build separated test cases to validate the optimized GEMV kernel, and ensure that the build process completes successfully without errors.
    - Remote run kernel test cases to validate the correctness and performance of the optimized GEMV kernel, target performance >85% of roofline for memory bound kernel, and confirm that the optimized kernel meets the performance requirements.
    - Remote build command: cd /mnt/river/moe/openvino && source build_release.sh
    - Remote e2e test environment setup: cd /mnt/river/moe/openvino.genai/ && source venv/bin/activate
    - Remote e2e test command: cd /mnt/river/moe/openvino.genai/tools/llm_bench && source runme.sh
        - Remote e2e test model: /mnt/river/models/qwen3-8b/pytorch/ov/OV_FP16-4BIT_DEFAULT
        - baseline test: export OV_GPU_MULTI_IMPL_POLICY="NONE"
        - optimized test: export OV_GPU_MULTI_IMPL_POLICY="AUTO_HEURISTIC"
    - Ensure that the remote machine has the necessary environment and dependencies set up for building and testing the GEMV kernel.
    - Monitor the remote build and test results, and analyze any failures or performance regressions.
9. **Document optimization**:
    - Summarize the optimizations applied and their impact on performance.
    - Include before-and-after performance metrics to demonstrate improvements.
    - Include the ratio to hardware roofline to show efficiency gains.

Keep explanations conversational. For complex concepts, use multiple analogies.