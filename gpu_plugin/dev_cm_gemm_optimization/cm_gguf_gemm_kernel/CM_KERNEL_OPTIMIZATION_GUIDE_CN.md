# C for Metal（CM）Kernel 使用与优化指南

> 本文根据本目录中的 `cmlangspec`、`cmcuserguide` 和 `cmoclrt` 整理，目标是为 CM kernel 的编写、编译、分析和性能优化提供一份可直接使用的手册。
>
> CM 是面向 Intel Gen/Xe GPU 的 C++ 方言。它描述的是一个硬件线程（hardware thread）的 SIMD 计算，而不是 OpenCL 中的单个 work-item。具体平台、驱动和 cmc 版本可能对某些指令和参数有额外限制；最终应以编译器诊断和目标平台文档为准。

## 1. 快速决策

优化一个 CM kernel 时，建议按下面顺序处理：

1. **确认目标平台和执行模型**：确定 `-march`、SIMD 宽度、线程组布局以及是否存在 Xe/XMX、LSC、SLM 等硬件能力。
2. **先保证数据布局正确**：选择合适的 `vector`/`matrix` 形状，使加载、计算和存储的 SIMD 宽度一致；避免不必要的重排、`iselect` 和标量循环。
3. **分析内存瓶颈**：连续访问优先用 block load/store；不连续访问再考虑 gather/scatter；重复使用的数据考虑 SLM，但必须核算 SLM 搬运、barrier 和寄存器成本。
4. **分析计算瓶颈**：整数矩阵乘优先考虑 `cm_dpas`/`cm_dpasw`，并使用 `CM_HAS_*` 能力宏；普通整数点积可检查 `cm_dp4a` 支持情况。
5. **控制寄存器压力**：观察寄存器使用量和最终 ISA；增大向量、双 GRF、展开循环或增加临时变量都可能降低 occupancy。
6. **用生成代码和实测结果闭环**：检查编译器报告、ISA、寄存器用量、内存消息和实际 kernel 时间，不能仅凭源代码推断性能。

---

## 2. CM 的执行模型

### 2.1 Kernel 与函数

Kernel 入口使用 `_GENX_MAIN_`（通常由头文件定义为 `__declspec(genx_main)`）：

```cpp
_GENX_MAIN_ void add_kernel(SurfaceIndex src, SurfaceIndex dst) {
    // kernel body
}
```

- 一个源文件可以包含多个 kernel 入口。
- `_GENX_MAIN_` kernel 不能被另一个 kernel 调用。
- `_GENX_` 用于设备端辅助函数；入口及其传递调用的 `_GENX_` 函数必须在同一个源文件中。
- CM kernel 不支持递归。
- 设备函数中避免使用复杂 C++ 特性；规范明确限制指针、C++ 引用、普通数组、继承、动态内存、异常、RTTI、静态变量等。

### 2.2 向量、矩阵和 SIMD

```cpp
vector<float, 16> v;
matrix<half, 8, 16> m;
vector_ref<float, 8> r = v.select<8, 1>(0);
```

- `vector<T, N>` 是长度为 `N` 的向量。
- `matrix<T, R, C>` 是 `R x C` 的矩阵，按 row-major 线性布局处理。
- `vector_ref`/`matrix_ref` 是对已有对象局部区域的引用，不额外拥有存储。
- 向量或矩阵总大小必须适合目标平台单个硬件线程可用的 GRF。
- 标量参与向量/矩阵运算时会按需要广播；同样元素数但形状不同的对象可以按 row-major 赋值或运算。
- 显式转换类型和形状，避免隐式提升导致额外指令或错误精度。

常用操作：

| 操作 | 用途 |
|---|---|
| `v(i)`、`m(i,j)` | 标量元素访问 |
| `select<N,S>(i)` | 向量区域选择，返回引用 |
| `select<R,VS,C,HS>(i,j)` | 矩阵区域选择，返回引用 |
| `select_all()` | 整个对象的引用 |
| `row(i)`、`column(i)` | 行/列引用 |
| `replicate<REP,...>()` | 复制/重排连续或带步长区域 |
| `iselect(idx)` | 按索引间接选择，返回新对象，不能作为左值 |
| `format<T,...>()` | 对连续、对齐的对象重解释形状/元素类型 |
| `merge(x, mask)` | 按 mask 更新向量/矩阵 |
| `any()`、`all()` | mask 布尔归约 |

`select` 的步长必须大于 0；当选取尺寸为 1 时，对应步长必须为 1。规范允许在明确不影响最终结果时使用越界 select，但越界元素是 don't-care，不能参与最终正确性。

### 2.3 标量控制流与 SIMD 控制流

- 普通 `if`、`for`、`while` 等要求条件为标量，形成所有 SIMD channel 一致的 uniform 控制流。
- 需要每个 channel 独立执行时使用：
  - `SIMD_IF_BEGIN` / `SIMD_ELSE` / `SIMD_ELSEIF` / `SIMD_IF_END`
  - `SIMD_DO_WHILE_BEGIN` / `SIMD_DO_WHILE_END`
  - `SIMD_BREAK` / `SIMD_CONTINUE`

SIMD 控制流宽度必须是大于 1 且不超过 32 的 2 的幂；嵌套块必须保持相同宽度。SIMD 块中不能使用普通标量控制流、普通 `break/continue` 或用户定义函数调用。block read/write、sampler、VME、线程通信和 dot/SAD/line 类操作也不能放入 SIMD 控制流；scatter read/write 仅在数据宽度与 SIMD 宽度匹配时可用。

**优化建议：**尽量保持主循环 uniform。分支发散会降低有效 SIMD 利用率；若必须发散，使用向量 mask、`merge` 和合法的 SIMD 控制流，并验证生成 ISA。

---

## 3. 类型、对齐和数据表示

### 3.1 支持的主要类型

标量包括 `char/uchar`、`short/ushort`、`int/uint`、`float`、`half`；部分平台还支持 `double`、`long long`、`svmptr_t`、`tfloat32`。能力与限制依赖目标平台：

- `double`：Gen7+，可用操作和内存消息受限。
- `long long`：Gen8+，不支持乘除及许多 intrinsic。
- `svmptr_t`：SVM 指针大小由 `/DCM_PTRSIZE=32` 或 `/DCM_PTRSIZE=64` 决定，必须与 host 指针大小一致。
- `tfloat32`：Gen12+，用于需要 TF32 的接口。

不支持的类型会在编译期报错。饱和运算通常通过 intrinsic 的 `SAT` 形式获得；浮点饱和范围为 `[0,1]`，整数饱和范围为目标类型可表示范围。

### 3.2 全局变量和 volatile

CM 全局向量/矩阵变量本质上是每个硬件线程私有的，不是线程组共享内存。`_GENX_VOLATILE_`（或 `__declspec(genx_volatile)`）可限制相关优化，常用于降低寄存器压力；但它也可能减少优化机会，应通过 ISA 和实测确认收益。

全局变量不能用普通初始化语法初始化。固定绑定偏移的 volatile global 必须由程序员保证不与其他绑定或 thread payload 重叠。

### 3.3 通过引用传参的代价

`vector_ref`/`matrix_ref` 参数使用寄存器间接访问，通常限制为 SIMD8，并带来地址计算和潜在延迟。非连续或非 whole-GRF-aligned 参数还可能触发 copy-in/copy-out。

实践规则：

- 不超过约两个 GRF 的只读参数，通常优先按值传递。
- OUT 参数尽量在辅助函数末尾一次性写回。
- 参数频繁访问、非连续或经常非整 GRF 对齐时，谨慎使用 pass-by-reference。
- 大参数、调用点多且参数整 GRF 对齐时，reference 可能有收益。

---

## 4. 内存访问

### 4.1 SurfaceIndex 和访问约束

`SurfaceIndex`、`SamplerIndex`、`VmeIndex` 是由 host/runtime 管理的 opaque handle，通常只能作为 kernel 参数或 intrinsic 参数使用。不要让不同 `SurfaceIndex` 发生 alias；typed 与 untyped 访问的排序可能无法保证。

CM 的内存访问由程序员声明语义：buffer、1D/2D/3D image、media block、sampler、SVM 等必须和实际资源类型匹配。错误的资源类型或错误的 intrinsic 属于未定义行为。

在 OpenCL/Level Zero runtime 模式下，资源参数要使用类型属性，例如：

```cpp
[[type("buffer_t")]] SurfaceIndex buf;
[[type("image2d_t read_only")]] SurfaceIndex img;
```

如果需要 SVM 指针，使用 `svmptr_t`；sampler 使用 `sampler_t`。旧语义下 `image2d_t` 默认可能按 media block 处理；普通 2D image 应使用 `-vc-use-plain-2d-images`，或明确使用 `image2d_media_block_t`。

### 4.2 Block load/store、gather/scatter

连续、对齐、规则访问优先使用 block 操作；随机访问才使用 gather/scatter。LSC 接口在目标支持 `CM_HAS_LSC` 时可用。

典型 LSC buffer block load/store：

```cpp
vector<uint, 16> x;
x = cm_load<uint, 16>(src, byte_offset);
cm_store<uint, 16>(dst, byte_offset, x);
```

关键规则：

- block 元素数只能是 `1, 2, 3, 4, 8, 16, 32, 64`。
- offset 是**字节偏移**，必须按数据尺寸对齐。
- gather/scatter 的 offset 也是字节偏移；`VS` 表示每个 offset 对应的元素数，结果总元素数必须满足 `M = N x VS`。
- 支持 `U8/U16/U32/U64` 的数据尺寸（具体 API 以目标编译器为准）。
- 越界 load 返回 0；越界 store 不更新内存。
- 使用 predicate 屏蔽不需要的 lane，避免无效访问。

LSC cache hint 必须使用合法组合。load/prefetch 常见策略包括 `Default`、`Cached`、`Uncached`、`Streaming`；store 包括 `WriteBack`、`WriteThrough`、`Uncached`、`Streaming`。不要猜测任意 L1/L2 组合，非法组合会编译失败或不符合目标硬件语义。

**访问优化原则：**

1. 让相邻 SIMD lane 访问相邻 cache line/字节区间。
2. 合并小访问，优先使用较宽的对齐 block message。
3. 避免多个 lane 重复读取同一块元数据；若必须复用，评估 SLM 或寄存器广播。
4. cache hint 只在已知访问模式下使用，并以 profile 验证，Streaming 并不总是优于 Cached。
5. load/store 的 vector 形状、对齐和边界处理应在编译期尽量确定。

### 4.3 传统 dataport 与模板库

规范还提供传统 block、scatter/gather、atomic、2D read/write 等接口。对于常见图像/线性 buffer I/O，优先查看 `cm/cmtl.h` 中的 `cmtl::ReadBlock`、`WriteBlock`、`ReadLinear`、`WriteLinear`。CM Template Library 的实现被设计为适合 CM，通常比手写等价代码更容易生成高效指令，并能适配未来变化。

---

## 5. SLM（Shared Local Memory）

SLM 是线程组内共享的高带宽存储，不由 system memory backing；内容初始未定义，生命周期随 group SLM 分配而定。每个 group 独立拥有 SLM，其他 group 不能直接访问。

### 5.1 基本流程

```cpp
_GENX_MAIN_ void slm_kernel(SurfaceIndex src, SurfaceIndex dst) {
    cm_slm_init(16 * 1024);             // 每 group 的 SLM 总大小（字节）
    uint tile = cm_slm_alloc(4096);     // 分配一个 SLM buffer，返回 byte offset

    // 将全局数据装入 tile；必要时再 cm_barrier()
    cm_slm_load(tile, src, 0, 1024);    // loadSize 必须是 256 的倍数

    vector<uint, 16> addr;
    vector<uint, 16> val;
    // addr 通常为元素偏移；scaled 版本使用字节偏移
    cm_slm_read(tile, addr, val);
    // 使用 val 计算
    cm_slm_write(tile, addr, val);
}
```

常用 ID API：`cm_local_id`、`cm_local_size`、`cm_group_id`、`cm_group_count`，以及对应的 linear 版本。它们用于将线程分工映射到 tile、group 和全局输出。

### 5.2 barrier、fence 与可见性

- 一个线程写入 SLM 后，其他线程要读取该数据，必须使用 `cm_barrier()`。
- Gen10+ 在 barrier 前需要保证读写顺序时，按规范加入 `cm_slm_fence(CM_GLOBAL_COHERENT_FENCE)`。
- `cm_slm_load` 会透明地分配内存读取工作，并插入使数据对 group 可见所需的 barrier；只为搬运全局数据到 SLM 时优先考虑它。
- `cm_global_barrier()` 只能用于 cooperative kernel；普通 kernel 使用属于未定义行为。
- `cm_sbarrier(1)` / `cm_sbarrier(0)` 可将 barrier 拆成 signal/wait，但参数必须是编译期常量。
- barrier 必须由相关线程以一致方式执行，不能让部分线程永久跳过导致死锁。

### 5.3 性能要点

- 优先使用 `cm_slm_read4`/`cm_slm_write4`，它们在早期 Gen 平台上带宽显著高于单元素版本。
- read4/write4 以 4 个 dword channel 进行访问，数据在硬件中可能是 transpose 布局；设计 tile 时尽量直接利用该布局，避免额外转置。
- `cm_slm_block_read/write` 要满足对齐：普通 block 通常需要 16-byte 对齐；dword-aligned 形式可能在硬件不支持时退化为 gather。
- SLM 总大小由 `cm_slm_init` 指定，早期 Gen 文档给出的每 group 上限为 64 KiB；实际目标使用 `CM_MAX_SLM_SIZE` 检查。
- 计算 SLM 收益时要同时计入 global-to-SLM 搬运、SLM 读写、barrier/fence 和额外地址计算；只减少 global load 不代表一定更快。
- 避免 SLM bank 冲突、重叠写和无必要的来回搬运。重叠写的顺序未定义。

---

## 6. DPAS/XMX 与专用计算

Xe 平台可用 `cm_dpas`/`cm_dpasw` 执行矩阵乘加：

$$D_{M\times N}=A_{M\times K}\times B_{K\times N}+C_{M\times N}$$

支持的 source precision 包括 2/4/8-bit 有符号或无符号整数、BF16、FP16 和 TF32，具体组合由 `CM_HAS_DPAS_*`、`CM_HAS_DPAS_ACC_*`、`CM_HAS_TF32` 等宏决定。

使用前检查：

- `SystolicDepth` 必须为 8。
- `RepeatCount` 为 1 到 8。
- `Src1` 使用 VNNI-packed 格式，`Src2` 使用 row-major 格式。
- `Src1Size`、`Src2Size`、accumulator 大小必须与模板参数和精度严格匹配。
- `cm_dpas` 目标依赖 `CM_HAS_DPAS`。
- `cm_dpasw` 只支持整数和 16-bit 浮点；由于 Source2 在两个 fused EU thread 间共享，不要用于部分 fused thread group、奇数线程组或 divergent code，否则行为未定义。

整数点积还可检查 `CM_HAS_DP4A` 和对应 `cm_dp4a` intrinsic。不要为了使用 DPAS 强行改变布局；布局转换成本、额外 load 和寄存器压力可能抵消算力收益。

---

## 7. 编译、目标选择和调试输出

### 7.1 目标平台

使用 `-march=<target>` 选择 Gen/Xe 目标，例如 `-march=SKL`、`-march=TGLLP`、`-march=DG2`、`-march=PVC`、`-march=PTL`。大小写不敏感。

推荐在源码中使用：

```cpp
#if defined(CM_GENX) && CM_GENX >= 1300
    // Xe3/PTL 及后续目标
#endif

#if defined(CM_HAS_DPAS)
    // 使用 cm_dpas
#else
    // 兼容实现
#endif
```

规范建议优先使用 `CM_GENX` 和 `CM_GENX_REVID`，而不是依赖即将弃用的 `CM_GEN12` 等平台宏。常见目标包括 GEN7_5/HSW、GEN8/BDW、GEN9/SKL、GEN11/ICLLP、GEN12/TGLLP、XeHPG/DG2、XeLPG/MTL、XeHPC/PVC、Xe2/BMG/LNL、Xe3/PTL。

### 7.2 常用 cmc 选项

| 选项 | 用途 |
|---|---|
| `-march=<gen>` | 选择 Gen/Xe 目标 |
| `-binary-format <cm\|ocl\|ze>` | 选择输出二进制格式 |
| `-emit-spirv` | 输出 SPIR-V |
| `-fcmocl` | 面向 OpenCL runtime 的编译 |
| `-g`、`-g1`、`-g2` | 调试信息；`-g1` 主要为行号，`-g2` 为完整调试信息 |
| `-mCM_printregusage` | 打印每个 kernel 的寄存器使用量 |
| `-Qxcm_print_asm_count` | 打印指令计数 |
| `-Qxcm_opt_report` | 输出 GenX Finalizer 优化报告 |
| `-mCM_printfargs` | 打印传给 finalizer 的参数 |
| `-mdump_asm` | 请求 assembly dump；推荐优先使用 shader dump |
| `-mCM_jit_option=<value>` | 传递 GenX Finalizer 选项 |
| `-Qxcm_register_file_size=128\|256\|auto` | 设置寄存器文件大小；XeHP+ 支持 128/256/auto |
| `-Qxcm_doubleGRF` | `-Qxcm_register_file_size=256` 的别名 |
| `-menableiga` | 使用 IGA 汇编语法 |
| `-mCM_no_vector_decomposition` | 禁止大向量拆分，需实测使用 |
| `-mCM_collect_cost_info` | 收集循环成本信息 |
| `-###` | 仅打印将执行的编译命令 |

所有 CM 专用选项通常接受 `-` 或 `/` 前缀；普通 Clang 选项不一定接受 `/`。

### 7.3 常用环境变量

- `ENABLE_IGA=1`：启用 IGA 语法，等价于 `-menableiga`。
- `CM_FORCE_ASSEMBLY_DUMP`：启用旧式 assembly dump 选项。
- `CM_INCLUDE_DIR`：指定 CM 头文件目录。
- `IGC_ShaderDumpEnable=1`：将 LLVM、assembly、ISA 输出到 `/tmp/IntelIGC/<application_name>`。
- `IGC_DumpToCurrentDir=1`：让 shader dump 写入当前目录。

调试构建时建议打开 `-g -mCM_printregusage -Qxcm_print_asm_count`，优化对比时保留同一目标、同一 finalizer 选项和同一 runtime 环境。

### 7.4 最小离线编译示例

```text
cmc kernel.cpp -march=SKL -m64 -fcmocl
cmc kernel.cpp -march=SKL -m64 -fcmocl -emit-spirv
```

如果 cmc 没有自动找到头文件，可按实际安装位置增加 `-isystem <path-to-cm-header>`。用 `-###` 先确认驱动最终调用的 front-end、GenX backend 和 finalizer 参数。

### 7.5 OpenCL/Level Zero runtime

- OpenCL 从 CM 源码构建时使用 `-cmc`；从 SPIR-V 构建时使用 `-vc-codegen`。
- Level Zero 从 SPIR-V 或预编译 binary 构建时使用 `-vc-codegen`。
- 可用 `ocloc` 离线生成 OpenCL binary；CM 源码使用 `-cmc`，SPIR-V 使用 `-vc-codegen`。
- CM 描述一个硬件线程，而 OpenCL runtime 通常把多个 work-item 打包进硬件线程。因此 host 的 local size、CM 线程数和 thread/group space 必须一致。
- CM runtime 不支持 global work offset。
- CM surface 大小不能超过 4 GB。

---

## 8. 循环、展开和寄存器压力

循环可使用：

```cpp
#pragma unroll(4)
for (uint i = 0; i < N; ++i) {
    ...
}
```

或：

```cpp
#pragma unroll
for (...) {
    ...
}
```

这是对 compiler 的请求，不是绝对保证；不可行时编译器可以不展开。展开通常能消除分支、暴露 ILP，但会增加代码体积、临时变量和寄存器压力。优化时按以下顺序尝试：

1. 先固定边界、对齐和步长，让编译器可分析。
2. 只展开短小、热点循环。
3. 对比未展开、部分展开、完全展开的寄存器数和 kernel 时间。
4. 若出现 spill、occupancy 下降或 instruction cache 压力，回退展开因子。

CM 的表达式遵循“先读取所有 operand，再执行操作”的语义；利用这一点可安全地使用同一对象的相邻 select 做表达式，但不要依赖未定义的别名或重叠写顺序。

---

## 9. Inline vISA Assembly

当 CM intrinsic 无法表达某个目标指令时，可在函数内部使用 GNU 风格 inline assembly：

```cpp
asm("add (M1, 8) %0 %1 %2"
    : "=r"(dst)
    : "r"(src0), "r"(src1));
```

常用约束：

| 约束 | 含义 |
|---|---|
| `r` | 普通寄存器变量，编译器推导 region |
| `rw` | 普通变量，汇编字符串自行提供 region |
| `a` | 间接地址访问 |
| `cr` | predicate |
| `i` | immediate |
| `F` | 浮点 immediate |
| `n` | 编译期已知常量，常用于执行宽度 |
| `=` | 只写输出 |
| `+` | 读写 operand |
| `0`、`1` 等 | 与对应输出 operand 绑定 |

注意事项：

- 编译器不解析 vISA 指令本身，不会替你验证 opcode、消息描述符、payload 大小、对齐或寄存器类型。
- 输入-only operand 不得被汇编修改；读写变量应使用 `+`，并在汇编字符串中分离 source/destination 位置。
- 能使用 `r` 自动推导 region 时优先使用 `r`，手写 `rw` 更容易造成越界或错误 stride。
- inline asm 不支持 clobber、asm goto、sampler/surface variable 显式操作。
- 每次改动后必须检查最终 VISA/ISA 和运行结果，inline asm 的错误可能表现为编译成功但执行错误。

---

## 10. 推荐的 kernel 优化流程

### 阶段 A：建立正确 baseline

- 固定 `-march`、编译器版本、driver、runtime、输入形状和线程/组配置。
- 记录 kernel 总时间、平均/尾部延迟、吞吐、内存带宽、正确性误差。
- 保存原始 ISA、寄存器用量、指令数、SLM 使用量和编译选项。
- 先确认结果 bit-exact 或满足明确的误差预算。

### 阶段 B：确定瓶颈

- **带宽受限**：观察实际 load/store 字节数、cache 命中和访问合并程度；优先优化布局、block message 和重复读取。
- **延迟/地址受限**：减少小消息、间接访问、标量地址计算和过细粒度 gather。
- **计算受限**：检查 DPAS/DP4A 是否能匹配数据类型和布局，减少无效转换和重复解码。
- **寄存器/occupancy 受限**：降低向量 live range、减少展开和临时 accumulator，比较 128/256 GRF 的代价。
- **发散受限**：减少 divergent path，用 mask/merge 或重新分工。

### 阶段 C：一次只改一个变量

每个实验只改变一个主要因素，例如 SIMD 宽度、block 大小、展开因子、SLM staging、cache hint 或 DPAS 路径。每次记录：

- 编译是否成功、目标能力宏是否符合预期。
- ISA 中的消息类型、执行宽度和循环结构。
- GRF/寄存器、spill、指令数、SLM 和 barrier 数量。
- 热点 kernel 时间和端到端时间。
- 输出误差、边界输入、随机输入和压力输入结果。

### 阶段 D：确认收益可迁移

在至少一个目标平台和一个兼容 fallback 上验证。平台特化代码应由 `CM_GENX`、`CM_GENX_REVID` 或 `CM_HAS_*` 宏隔离，并保留正确的通用路径。不要把某一机器上的 cache 行为、频率或 runtime 调度直接当成普遍规律。

---

## 11. 常见错误与排查表

| 现象 | 优先检查 |
|---|---|
| intrinsic 找不到 | `-march` 是否正确、`CM_HAS_*` 是否定义、头文件版本是否匹配 |
| block load/store 编译失败 | 元素数、数据尺寸、offset 对齐、cache hint 组合 |
| 结果偶发错误 | SLM barrier/fence、重叠写、SurfaceIndex alias、越界 select、mask 语义 |
| 端到端变慢但 kernel 变快 | 寄存器压力、occupancy、搬运/barrier 开销、host dispatch 和其他 kernel |
| SIMD 版本不如标量版本 | 控制流发散、SIMD 宽度不匹配、无效 lane、额外重排 |
| DPAS 结果错误 | VNNI/source layout、precision、`SystolicDepth=8`、尺寸模板参数 |
| cm_slm_read/write 很慢 | 是否可改用 read4/write4 或 block API，是否产生了额外 transpose |
| 编译成功但 inline asm 错误 | opcode、执行宽度、region、payload、约束和目标 ISA |
| 调试文件找不到 | `IGC_ShaderDumpEnable=1`、`IGC_DumpToCurrentDir=1` 和实际工作目录 |
| host 启动失败 | resource type attribute、CM/OpenCL/Level Zero 编译选项、thread/group space、4 GB 限制 |

---

## 12. 最终检查清单

- [ ] `-march` 与实际运行 GPU 一致，必要时记录 `CM_GENX_REVID`。
- [ ] 所有平台特性都有 `CM_HAS_*` 宏保护和 fallback。
- [ ] 向量/矩阵总大小、形状、对齐和 select stride 合法。
- [ ] Surface 类型属性与真实资源类型一致，未发生不允许的 alias。
- [ ] block/gather/scatter 的 offset、数据尺寸、元素数和 predicate 正确。
- [ ] SLM 已初始化、分配不超限；跨线程读写使用正确 barrier/fence。
- [ ] barrier 路径对所有相关线程一致，不会死锁。
- [ ] DPAS/DP4A 的精度、布局和尺寸满足目标硬件约束。
- [ ] 循环展开、向量化和双 GRF 的收益已用 ISA、寄存器和性能数据验证。
- [ ] 正确性覆盖边界、越界、随机、不同尺寸和 fallback 路径。
- [ ] 性能结果同时包含单 kernel 和端到端指标，并在清理/固定缓存状态后复测。

## 13. 本目录中的原始参考

- `_sources/cmlangspec/cmlangspec.rst.txt`：语言、类型、向量/矩阵、控制流、intrinsic、SLM、XMX、inline asm 和模板库规范。
- `_sources/cmcuserguide/cmcuserguide.rst.txt`：cmc 目标平台、编译选项、隐式宏和环境变量。
- `_sources/cmoclrt/cmocl.rst.txt`：OpenCL/Level Zero kernel 与 host 集成、资源属性、线程模型、ocloc 和 runtime 选项。
- `cmlangspec/cmlangspec.html`、`cmcuserguide/cmcuserguide.html`、`cmoclrt/cmocl.html`：本目录提供的 HTML 版本。
