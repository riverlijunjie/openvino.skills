/*
[USER_INSTRUCTIONS_START]
Consider some kernel optimizations tips :
    - gemm is compute bound, focus on improving FLOPS and reducing or hiding memory access latency.
    - Use local memory to reduce global memory access latency if possible.
    - Optimize work-group sizes based on the GPU architecture.
    - Minimize divergent branches within the kernel.
    - Unroll loops where applicable to increase instruction-level parallelism.
    - Use vectorized data types and block read to improve memory bandwidth utilization.
    - Optimize memory access patterns to ensure coalesced accesses, which can significantly improve performance.
    - Consider tiling techniques to maximize data reuse in local memory.
    - Must use fused dpas(XMX) operations:
        float  intel_sub_group_f16_f16_matrix_mad_k16(int  a, int8 b, float  acc);
        float2 intel_sub_group_f16_f16_matrix_mad_k16(int2 a, int8 b, float2 acc);
        float4 intel_sub_group_f16_f16_matrix_mad_k16(int4 a, int8 b, float4 acc);
        float8 intel_sub_group_f16_f16_matrix_mad_k16(int8 a, int8 b, float8 acc);

            This section describes a family of built-in functions that multiply two matrix sources a and b and then add a matrix accumulation value to produce a matrix result value. a is the first matrix operand and has M rows and K columns. b is the second matrix operand and has K rows and N columns. acc is the matrix accumulation value and has M rows and N columns. The result value also has M rows and N columns. All work items in the subgroup cooperate to perform this operation. These functions must be encountered by all work items in the subgroup executing the kernel.
            The dimensions of the two source matrices and the elements of each source matrix are described by the built-in function name and its arguments.
            As an example, given the function:
            int2 intel_sub_group_u8_i8_matrix_mad_k32(uint2 a, int8  b, int2 acc);
            a is the first source matrix operand and has M rows and K columns.
            The value for M is determined by the number of vector components in the source operand a. In the example above, a is a uint2 argument, therefore the matrix a operand has M equal to 2 rows.
            The value of K is described by the function name. In this case, the value of K is 32, therefore the matrix a operand has K equal to 32 columns.
            The matrix component data type is also described by the function name. In this case, the matrix a component data type is u8, indicating that the elements of the matrix a operand are unsigned 8-bit integers.
            Each work item contributes part of this matrix. In this case, since the elements of the matrix a are 8-bit integers, and since each work item is contributing 32 bits (the size of a uint) of data per row of this matrix, each work item is contributing four 8-bit integer values per row.
            Since K is 32, and each work item is contributing four 8-bit values per row, the number of work items in the subgroup must be equal to 8.
            b is the second source matrix operand and has K rows and N columns.
            Each work item contributes one column of this matrix. Therefore, the number of columns N is equivalent to the subgroup size.
            As above, the value of K is described by the function name. In this case, the value of K is 32, therefore the matrix b operand has K equal to 32 rows.
            As above, the matrix component data type is described by the function name. In this case, the matrix b component data type is i8, indicating that the elements of the matrix b operand are signed 8-bit integers.
            Since K is 32 and the elements of the matrix b are 8-bit integers, each work item must contribute 256 bits of source data to contribute K values. The 256 bits of source data are packed and passed as the int8 argument b.
            acc specifies the accumulation value and has M rows and N columns.
            As above, the value of M is determined by the number of components in the source operand acc. In the example above, acc is an int2 argument, therefore the accumulation value operand has M equal to 2 rows.
            Since both a and acc specify operands with M rows, and since the value of M is determined by the number of components in the source operand, both the a and acc operands will be vector operands with the same number of components.
            As above, each work item contributes one column of accumulation values. Therefore, the number of columns N is equivalent to the subgroup size.
            The acc operand is a "full precision" accumulation value. In the example above, the matrices contain integer data, therefore the acc operand is a vector of int data.
            The result value returned by the function also has M rows and N columns.
            As above, the value of M is determined by the number of components in the return type. In the example above, the return type is int2, therefore the result value has M equal to 2 rows.
            Since the result value, a, and acc all specify values with M rows, and since the value of M is determined by the number of components in the source operand or return type, the return tye, a, and acc will all be vectors with the same number of components.
            As above, each work item will receive one column of result values. Therefore, the number of columns N is equivalent to the subgroup size.
            Similar to the acc operand, the return value is a "full precision" result value. In the example above, the matrices contain integer data, therefore the return type is a vector of int data.
            The full list of supported functions is described in the overview, above. For this list of functions:
            M may be equal to 1, 2, 4, or 8.
            N must be equal to 8 for some devices or 16 for other devices. In other words, the only supported subgroup sizes are 8 and 16.
            Supported integer matrix types for a and b are any combination of signed or unsigned 8-bit integers, or any combination of signed or unsigned 4-bit integers. For 8-bit matrices, K must be equal to 32. For 4-bit matrices, K must be equal to 64. For these integer matrix types, the accumulation value acc and result value are signed 32-bit integers.
            The supported floating-point matrix types for a and b are fp16 (half) or bfloat16. For these floating-point matrices, K must be equal to 16. The accumulation value acc and result value are 32-bit floating-point values. For devices with N equal to 16, the accumulation value acc and result value may also be fp16 for fp16 matrices, or bfloat16 for bfloat16 matrices.

        How to use DPAS in OpenCL:
            1. Enable Extensions
            #pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
            #pragma OPENCL EXTENSION cl_intel_subgroups : enable
            #pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable

            2. Set Subgroup Size
            - Add: __attribute__((intel_reqd_sub_group_size(N))) to the kernel
            - N = 8 or 16, must match the device's minimum subgroup size

            3. Common Function Signatures
            - FP16:
            float8 intel_sub_group_f16_f16_matrix_mad_k16(short8 a, int8 b, float8 acc); // M=8, N=16
            - INT8:
            int8 intel_sub_group_i8_i8_matrix_mad_k32(int8 a, int8 b, int8 acc); // M=8, N=8

            4. Data Packing Rules
            - Matrix A: Each work item holds M rows, each row packed as short (N=16) or int (N=8)
            - Matrix B: Each work item holds 1 column, K elements packed as int8
            - fp16 should be passed as signed types (use as_short/as_int)

            5. Data Loading
            - Recommended: use intel_sub_group_block_read8 to load B
            - Pointer must be 4-byte aligned

            6. Usage Example
            float8 acc = (float8)(0.0f);
            float8 result = intel_sub_group_f16_f16_matrix_mad_k16(a_packed, b_packed, acc);

            7. Loop Accumulation
            - For K_total > K_TILE, call DPAS multiple times in a loop, accumulating the result

            8. Notes
            - All work items in the subgroup must participate in DPAS
            - Subgroup size must match the device
            - Only supported on Intel GPUs with this extension

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
