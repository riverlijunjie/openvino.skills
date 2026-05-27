
---
name: dev_study_cc
description: Develop kernel with claude code.
---

**Identify GPU architecture**: 
    - GPU architectures: Intel GPU PTL 12Xe, which is xe3 architecture.
    - Xe Core Number: 12
    - EU number of each Xe Core: 8
    - Threads number of each EU: 10
    - Subgroup size: 16 or 32
    - Register number of each EU: 256
    - Register size: 256 bytes
    - Shared Local memory size: 32KB

**Compute the theoretical roofline for the target GPU architecture based on its specifications (e.g., memory bandwidth, compute throughput)**:
    - Computation throughput can be calculated based on the number of EUs,and the clock frequency. For example:
      - FP16 XMX peak: 12 (Xe cores) × 8 (EUs/core) × 256 (FLOPs per cycle for FP16 XMX ) × 2400 MHz
      - INT8 XMX peak: 12 (Xe cores) × 8 (EUs/core) × 512 (FLOPs per cycle for INT8 XMX) × 2400 MHz

**Consider some kernel optimizations tips**:
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

**OpenCL API reference for DPAS instruction and related extensions**:

  - https://github.com/intel/intel-graphics-compiler/blob/6ecd10bc2d721e79d8efec39c62b740bded8aa95/documentation/visa/instructions/DPASW.md
  - https://registry.khronos.org/OpenCL/extensions/intel/cl_intel_subgroup_matrix_multiply_accumulate.html
  - https://registry.khronos.org/OpenCL/extensions/intel/cl_intel_subgroups_short.html
  - https://registry.khronos.org/OpenCL/extensions/intel/
  - https://registry.khronos.org/OpenCL/extensions/intel/cl_intel_subgroup_matrix_multiply_accumulate.html
  - https://registry.khronos.org/OpenCL/extensions/intel/cl_intel_subgroup_split_matrix_multiply_accumulate.html
  - https://registry.khronos.org/OpenCL/extensions/intel/cl_intel_subgroups.html
  - https://registry.khronos.org/OpenCL/extensions/intel/cl_intel_unified_shared_memory.html#_using_unified_shared_memory_with_kernels


Remote machine:
  - Target Remote Windows machine with PTL 12Xe GPU: 
      - Target hardware: PTL
      - GPU frequency: 2400 MHz
      - Local_Admin@10.239.132.229
      - password:openvino
      - openvino directory: C:\Users\Local_Admin\river\kernel_foundry\cc
      - cliloader is loacted in: C:\Users\Local_Admin\Downloads\clintercept-3.0.6-win64\Release\cliloader.exe
      - setup environment: cd C:\Users\Local_Admin\river\kernel_foundry && .\venv\Scripts\activate