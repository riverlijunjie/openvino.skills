# Intel OpenCL DPAS (Dot Product Accumulate Systolic) 使用指南

本文聚焦 `intel_sub_group_f16_f16_matrix_mad_k16` 和 `intel_sub_group_i8_i8_matrix_mad_k32` 两个内置函数的用法，基于 `cl_intel_subgroup_matrix_multiply_accumulate` 扩展。

---

## 1. 概述

DPAS 在 OpenCL 中通过 subgroup 协作完成矩阵乘加运算：

$$
\text{Result}_{M \times N} = A_{M \times K} \cdot B_{K \times N} + \text{Acc}_{M \times N}
$$

- **subgroup 中所有 work item 协作**完成一次 DPAS 调用
- 硬件在底层映射到 Xe 核心的 systolic array
- 这是低级接口，适合手写高性能 GEMM kernel

---

## 2. 所需扩展

```c
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable   // 用于 block_read_us
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable
```

---

## 3. 核心维度约束

| 参数 | 含义 | 取值 |
|------|------|------|
| **M** | 结果行数 (每个 work item 持有 M 行) | 1, 2, 4, 8 |
| **N** | 结果列数 = subgroup size | 8 或 16 |
| **K** | A 的列数 / B 的行数 | 取决于数据类型 |

**关键规则**：N = subgroup size。设备的最小 subgroup size 决定使用哪组函数签名。

---

## 4. `intel_sub_group_f16_f16_matrix_mad_k16` — FP16 矩阵乘加

### 4.1 语义

计算 `M×16` (fp16) × `16×N` (fp16) + `M×N` (fp32/fp16) → `M×N` (fp32/fp16)

K = 16，每两个 fp16 值 pack 进一个 32-bit (或 16-bit) 整型中。

### 4.2 函数签名

#### 设备最小 subgroup size = 8 (N=8)，float 累加器

```c
float  intel_sub_group_f16_f16_matrix_mad_k16(int  a, int8 b, float  acc);  // M=1
float2 intel_sub_group_f16_f16_matrix_mad_k16(int2 a, int8 b, float2 acc);  // M=2
float4 intel_sub_group_f16_f16_matrix_mad_k16(int4 a, int8 b, float4 acc);  // M=4
float8 intel_sub_group_f16_f16_matrix_mad_k16(int8 a, int8 b, float8 acc);  // M=8
```

#### 设备最小 subgroup size = 16 (N=16)，float 累加器

```c
float  intel_sub_group_f16_f16_matrix_mad_k16(short  a, int8 b, float  acc);  // M=1
float2 intel_sub_group_f16_f16_matrix_mad_k16(short2 a, int8 b, float2 acc);  // M=2
float4 intel_sub_group_f16_f16_matrix_mad_k16(short4 a, int8 b, float4 acc);  // M=4
float8 intel_sub_group_f16_f16_matrix_mad_k16(short8 a, int8 b, float8 acc);  // M=8
```

#### 设备最小 subgroup size = 16 (N=16)，half 累加器

```c
half   intel_sub_group_f16_f16_matrix_mad_k16(short  a, int8 b, half  acc);  // M=1
half2  intel_sub_group_f16_f16_matrix_mad_k16(short2 a, int8 b, half2 acc);  // M=2
half4  intel_sub_group_f16_f16_matrix_mad_k16(short4 a, int8 b, half4 acc);  // M=4
half8  intel_sub_group_f16_f16_matrix_mad_k16(short8 a, int8 b, half8 acc);  // M=8
```

### 4.3 参数含义

| 参数 | 含义 | 数据布局 |
|------|------|----------|
| `a` | 矩阵 A 数据 (M×K)，fp16 pack 在 int/short 中 | 每个 work item 持有 M 行，每行贡献 K/N 个 fp16 值 |
| `b` | 矩阵 B 数据 (K×N)，fp16 pack 在 int8 中 | 每个 work item 持有 1 列 (K 个 fp16 值 = 256 bits = int8) |
| `acc` | 累加值 (M×N) | 每个 work item 持有 1 列 M 个值 |
| 返回值 | 结果 (M×N) | 同 acc 布局 |

### 4.4 数据 Pack 规则

- **fp16 → int/short**: 2 个 fp16 值 pack 进一个 32-bit int（或 1 个 fp16 pack 进 16-bit short）
- 使用 `as_int()` / `as_short()` 进行 reinterpret cast
- **注意**：扩展使用 **signed** 类型表示 fp16 数据（即使逻辑上无符号，也用 int/short 而非 uint/ushort）

### 4.5 数据加载示例 (subgroup size=16)

```c
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void gemm_f16(...) {
    // 加载 A: 每个 work item 加载自己那份 fp16 数据
    // 对于 M=8, K=16, subgroup_size=16:
    //   每个 work item 贡献 K/16 = 1 个 fp16 值/行 → 1 short/行
    //   M=8 行 → short8 a
    short8 a_packed;
    for (int m = 0; m < 8; m++) {
        // 每个 work item 读 1 个 fp16 (即 1 个 ushort)
        ushort val = *((__global ushort*)A_ptr + row_offset + m * lda + get_sub_group_local_id());
        a_packed[m] = as_short(val);
    }

    // 加载 B: 每个 work item 加载 1 整列 (K=16 个 fp16 = 16 shorts = 8 ints)
    int8 b_packed = intel_sub_group_block_read8((__global uint*)(B_ptr + col_block_offset));

    // DPAS
    float8 acc = (float8)(0.0f);
    float8 result = intel_sub_group_f16_f16_matrix_mad_k16(a_packed, b_packed, acc);
}
```

---

## 5. `intel_sub_group_i8_i8_matrix_mad_k32` — INT8 矩阵乘加

### 5.1 语义

计算 `M×32` (int8) × `32×N` (int8) + `M×N` (int32) → `M×N` (int32)

K = 32，每 4 个 int8 值 pack 进一个 32-bit 整型。

### 5.2 函数签名

#### 设备最小 subgroup size = 8 (N=8)

```c
int  intel_sub_group_i8_i8_matrix_mad_k32(int   a, int8  b, int  acc);  // M=1
int2 intel_sub_group_i8_i8_matrix_mad_k32(int2  a, int8  b, int2 acc);  // M=2
int4 intel_sub_group_i8_i8_matrix_mad_k32(int4  a, int8  b, int4 acc);  // M=4
int8 intel_sub_group_i8_i8_matrix_mad_k32(int8  a, int8  b, int8 acc);  // M=8
```

#### 设备最小 subgroup size = 16 (N=16)

```c
int  intel_sub_group_i8_i8_matrix_mad_k32(short   a, int8  b, int  acc);  // M=1
int2 intel_sub_group_i8_i8_matrix_mad_k32(short2  a, int8  b, int2 acc);  // M=2
int4 intel_sub_group_i8_i8_matrix_mad_k32(short4  a, int8  b, int4 acc);  // M=4
int8 intel_sub_group_i8_i8_matrix_mad_k32(short8  a, int8  b, int8 acc);  // M=8
```

### 5.3 参数含义

| 参数 | 含义 | 数据布局 |
|------|------|----------|
| `a` | 矩阵 A (M×K)，int8 pack 在 int/short 中 | 每 work item 持 M 行；N=8 时每行 4 个 int8 (1 int)，N=16 时每行 2 个 int8 (1 short) |
| `b` | 矩阵 B (K×N)，int8 pack 在 int8 中 | 每 work item 持 1 列 K=32 个 int8 = 256 bits = int8 向量 |
| `acc` | 累加值 (M×N)，int32 | 每 work item 持 1 列 M 个 int32 |
| 返回值 | 结果 (M×N)，int32 | 同 acc |

### 5.4 数据 Pack 规则

- **subgroup size = 8**: 每 work item 每行贡献 32/8 = 4 个 int8 → pack 为 1 个 `int` (用 `as_int(char4(...))`)
- **subgroup size = 16**: 每 work item 每行贡献 32/16 = 2 个 int8 → pack 为 1 个 `short` (用 `as_short(char2(...))`)
- B 矩阵: 每 work item 贡献整列 K=32 个 int8 = 8 个 int (int8 向量)

### 5.5 功能等价伪码 (subgroup size=8, M=2)

```c
// 辅助: 4元素 dot product + accumulate
int dot4_acc(char4 a, char4 b, int acc) {
    return a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w + acc;
}

// 每一行的 vector-matrix multiply:
int vec_mat_mul_k32(int v, int8 b, int acc) {
    int result = acc;
    for (int i = 0; i < 8; i++) {  // 8 = subgroup_size
        result = dot4_acc(
            as_char4(sub_group_broadcast(v, i)),
            as_char4(b[i]),
            result);
    }
    return result;
}

// M=2 的完整函数:
int2 intel_sub_group_i8_i8_matrix_mad_k32(int2 a, int8 b, int2 acc) {
    int2 result;
    result.x = vec_mat_mul_k32(a.x, b, acc.x);
    result.y = vec_mat_mul_k32(a.y, b, acc.y);
    return result;
}
```

---

## 6. 完整 GEMM Kernel 示例 (FP16, subgroup size=16)

```c
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable

#define M_TILE 8   // DPAS M
#define K_TILE 16  // DPAS K for fp16
#define SG_SIZE 16 // subgroup size = N

__attribute__((intel_reqd_sub_group_size(SG_SIZE)))
__kernel void gemm_dpas_f16(
    __global half* A,   // [M_total x K_total], row-major
    __global half* B,   // [K_total x N_total], row-major
    __global float* C,  // [M_total x N_total], row-major
    int M_total, int N_total, int K_total)
{
    // 每个 workgroup 处理一个 tile
    int wg_m = get_group_id(0) * M_TILE;
    int wg_n = get_group_id(1) * SG_SIZE;
    int sg_lid = get_sub_group_local_id();

    float8 acc = (float8)(0.0f);  // M=8 rows of accumulator

    for (int k = 0; k < K_total; k += K_TILE) {
        // --- 加载 A tile: M_TILE x K_TILE ---
        // 每个 work item 加载 1 个 fp16/行 (因为 K/SG_SIZE = 16/16 = 1)
        // M=8 行 → short8
        short8 a_data;
        for (int m = 0; m < M_TILE; m++) {
            int row = wg_m + m;
            int col = k + sg_lid;
            ushort val = (row < M_total && col < K_total)
                ? as_ushort(A[row * K_total + col])
                : (ushort)0;
            ((short*)&a_data)[m] = as_short(val);
        }

        // --- 加载 B tile: K_TILE x SG_SIZE ---
        // 使用 block_read 加载 (每 work item 持有 1 列)
        // K=16 fp16 = 16 shorts = 8 ints → int8
        __global uint* B_uint = (__global uint*)(B + k * N_total + wg_n);
        int8 b_data = intel_sub_group_block_read8(B_uint);

        // --- DPAS ---
        acc = intel_sub_group_f16_f16_matrix_mad_k16(a_data, b_data, acc);
    }

    // --- 写回 C ---
    for (int m = 0; m < M_TILE; m++) {
        int row = wg_m + m;
        int col = wg_n + sg_lid;
        if (row < M_total && col < N_total) {
            C[row * N_total + col] = ((float*)&acc)[m];
        }
    }
}
```

---

## 7. 完整 GEMM Kernel 示例 (INT8, subgroup size=8)

```c
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable

#define M_TILE 8
#define K_TILE 32
#define SG_SIZE 8

__attribute__((intel_reqd_sub_group_size(SG_SIZE)))
__kernel void gemm_dpas_i8(
    __global char* A,    // [M_total x K_total], row-major, signed int8
    __global char* B,    // [K_total x N_total], row-major, signed int8
    __global int*  C,    // [M_total x N_total], row-major, int32
    int M_total, int N_total, int K_total)
{
    int wg_m = get_group_id(0) * M_TILE;
    int wg_n = get_group_id(1) * SG_SIZE;
    int sg_lid = get_sub_group_local_id();

    int8 acc = (int8)(0);

    for (int k = 0; k < K_total; k += K_TILE) {
        // --- 加载 A: M_TILE x K_TILE ---
        // subgroup_size=8, K=32: 每 work item/行贡献 32/8=4 bytes = 1 int
        int8 a_data;
        for (int m = 0; m < M_TILE; m++) {
            int row = wg_m + m;
            int base_col = k + sg_lid * 4;  // 每个 work item 负责 4 个连续 int8
            char4 a_chars;
            for (int i = 0; i < 4; i++) {
                int col = base_col + i;
                a_chars[i] = (row < M_total && col < K_total) ? A[row * K_total + col] : 0;
            }
            ((int*)&a_data)[m] = as_int(a_chars);
        }

        // --- 加载 B: K_TILE x SG_SIZE ---
        // 每 work item 持有 1 列, K=32 个 int8 = 32 bytes = 8 ints = int8
        __global uint* B_uint = (__global uint*)(B + k * N_total + wg_n);
        int8 b_data = intel_sub_group_block_read8(B_uint);

        // --- DPAS ---
        acc = intel_sub_group_i8_i8_matrix_mad_k32(a_data, (int8)b_data, acc);
    }

    // --- 写回 C ---
    for (int m = 0; m < M_TILE; m++) {
        int row = wg_m + m;
        int col = wg_n + sg_lid;
        if (row < M_total && col < N_total) {
            C[row * N_total + col] = ((int*)&acc)[m];
        }
    }
}
```

---

## 8. 数据加载最佳实践

### 8.1 使用 `intel_sub_group_block_read` 加载 B 矩阵

B 矩阵在 DPAS 中是 K×N 布局，每个 work item 持有一列。最佳加载方式：

```c
// 32-bit block read (适用于 A/B 都可以用)
uint8 data = intel_sub_group_block_read8((__global uint*)ptr);

// 16-bit block read (适用于 fp16 数据)
ushort8 data = intel_sub_group_block_read_us8((__global ushort*)ptr);
```

**对齐要求**：
- Block read: 指针必须 4-byte 对齐
- Block write: 指针必须 16-byte 对齐

### 8.2 加载 A 矩阵

A 矩阵的行分布在 subgroup 的各 work item 之间。通常有两种方式：
1. 每个 work item 直接按 stride 加载自己的份额
2. 使用 `sub_group_broadcast` 将一个 work item 的数据广播给其他人

---

## 9. Split Matrix Multiply (DPASW)

`cl_intel_subgroup_split_matrix_multiply_accumulate` 扩展提供跨两个 subgroup 协作的变体：

```c
float8 intel_sub_group_f16_f16_split_matrix_mad_k16(int4 a, int8 b, float8 acc); // M=8
int8   intel_sub_group_i8_i8_split_matrix_mad_k32(int4 a, int8 b, int8 acc);     // M=8
```

- 两个相邻 subgroup 各提供 A 矩阵的一半行（M/2 行）
- B 矩阵在两个 subgroup 间共享（减少 GRF 带宽需求）
- 仅支持 subgroup size = 8
- 对应硬件指令 DPASW（XeHP 特有，PVC 上不可用）

---

## 10. 其他数据类型变体

| 函数名 | A 类型 | B 类型 | Acc/结果类型 | K |
|--------|--------|--------|-------------|---|
| `intel_sub_group_i8_i8_matrix_mad_k32` | signed int8 | signed int8 | int32 | 32 |
| `intel_sub_group_i8_u8_matrix_mad_k32` | signed int8 | unsigned int8 | int32 | 32 |
| `intel_sub_group_u8_i8_matrix_mad_k32` | unsigned int8 | signed int8 | int32 | 32 |
| `intel_sub_group_u8_u8_matrix_mad_k32` | unsigned int8 | unsigned int8 | int32 | 32 |
| `intel_sub_group_f16_f16_matrix_mad_k16` | fp16 | fp16 | float32 或 half | 16 |
| `intel_sub_group_bf16_bf16_matrix_mad_k16` | bf16 | bf16 | float32 或 short(bf16) | 16 |
| `intel_sub_group_i4_i4_matrix_mad_k64` | signed int4 | signed int4 | int32 | 64 |

---

## 11. 关键注意事项

1. **必须设置 subgroup size**：使用 `__attribute__((intel_reqd_sub_group_size(N)))` 标注 kernel，N 必须等于设备最小 subgroup size（8 或 16）

2. **所有 work item 必须参与**：DPAS 是 subgroup 级协作操作，所有 work item 必须同时执行该函数

3. **fp16 用 signed 类型传递**：spec 规定 fp16 和 bf16 数据使用 signed int/short 传递（而非 unsigned），需用 `as_int()` / `as_short()` 做 reinterpret

4. **B 矩阵对齐**：使用 block_read 加载 B 时，指针必须 4-byte 对齐

5. **A 矩阵的 pack 方式因 subgroup size 而异**：
   - subgroup=8: A 每行用 int (4 bytes = K/8 个元素)
   - subgroup=16: A 每行用 short (2 bytes = K/16 个元素)

6. **迭代累加**：对于 K_total > K_TILE 的情况，循环多次 DPAS 调用，将上一次结果作为下一次的 acc

7. **设备查询**：使用 `clGetDeviceInfo` 查询 `CL_DEVICE_EXTENSIONS` 确认设备支持该扩展；使用 `clGetKernelSubGroupInfoKHR` 查询实际 subgroup size

8. **未定义行为**：在不支持的设备上调用、或 subgroup size 不匹配时行为未定义

---

## 12. 参考链接

- [cl_intel_subgroup_matrix_multiply_accumulate](https://registry.khronos.org/OpenCL/extensions/intel/cl_intel_subgroup_matrix_multiply_accumulate.html)
- [cl_intel_subgroup_split_matrix_multiply_accumulate](https://registry.khronos.org/OpenCL/extensions/intel/cl_intel_subgroup_split_matrix_multiply_accumulate.html)
- [cl_intel_subgroups](https://registry.khronos.org/OpenCL/extensions/intel/cl_intel_subgroups.html)
- [cl_intel_subgroups_short](https://registry.khronos.org/OpenCL/extensions/intel/cl_intel_subgroups_short.html)
- [IGC DPASW instruction](https://github.com/intel/intel-graphics-compiler/blob/master/documentation/visa/instructions/DPASW.md)
