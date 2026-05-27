# GEMM 内核优化总结

**目标硬件**: Intel Arc B390 (PTL 12Xe, Xe3 架构)  
**任务**: 对 GEMM (`C[M,N] = A[M,K] × B[K,N]`) 进行 OpenCL 深度优化  
**最终成果**: 相对 reference 加速 **153~217×**，达到 FP16 XMX 理论峰值的 **~45%**

---

## 一、硬件规格

| 参数 | 值 |
|---|---|
| 架构 | Xe3 (PTL 12Xe) |
| GPU | Intel Arc B390 |
| Xe-core 数 | 12 |
| EU / Xe-core | 8 |
| Thread / EU | 10（large-GRF 模式下 4） |
| Sub-group 大小 | SIMD16 |
| SLM / WG | 32 KB |
| GRF / thread（小） | 128 × 32 B = 4 KB |
| GRF / thread（大） | 256 × 32 B = 8 KB |
| 时钟频率 | 2400 MHz |
| **FP16 XMX 峰值** | **58.98 TFLOPS** |
| FP32 向量峰值 | 3.69 TFLOPS |

---

## 二、关键扩展与 DPAS 指令

使用的 OpenCL 扩展：
```
cl_intel_subgroups
cl_intel_subgroups_short
cl_intel_subgroup_matrix_multiply_accumulate
cl_intel_subgroup_local_block_io
cl_khr_fp16
```

DPAS 核心指令（FP16×FP16 → FP32 累加）：
```c
float8 intel_sub_group_f16_f16_matrix_mad_k16(short8 a, int8 b, float8 acc)
```
- 每次调用：8行 × 16列 输出（SIMD16 sub-group），消耗 K=16 列
- 每调用一次：8 × 16 × 16 × 2 = **4096 FLOPs**

Sub-group 宽块读：
```c
ushort8 intel_sub_group_block_read_us8(const __global/__local ushort* p)
```
- lane `lid` 读取 `p[i*16 + lid]`（i = 0..7）—— **行步长必须为 16**

---

## 三、优化迭代过程

### 起点（baseline）
- 简单 fp32→fp16 转换 + DPAS，无 SLM
- **9.62 TFLOPS**（16.31% 峰值）

---

### 迭代 1：SLM v1（标量读取）
- 引入 SLM tile 缓存 A/B，但用逐元素访问 SLM
- **4.60 TFLOPS**（回退）
- **原因**：SLM 非连续访问，无法触发宽读指令

### 迭代 2：SLM v2（local block_read）
- B_slm 改为 n-tile 步长 16 的布局，使用 `intel_sub_group_block_read_us8`
- **7.45 TFLOPS**

### 迭代 3：SLM + 双缓冲（Double Buffering）
- 尝试 ping-pong buffer 隐藏全局内存延迟
- **6.99 TFLOPS**（无收益，barrier 开销主导）

### 迭代 4：Large-GRF 模式
- 增加编译选项 `-cl-intel-256-GRF-per-thread`
- 16 个 `float8` 累加器 = 512 个 32-bit 寄存器 = 256 GRF 行，超过小 GRF 模式上限
- **9.16 TFLOPS**（**+2× vs 无此选项**）
- **结论**：使用 16 个 float8 累加器时必须开启大 GRF 模式

### 迭代 5：K_TILE=32（SLM 双条带）
- 每次外循环处理 32 个 K，减少 barrier 次数
- A_slm 布局：`[strip][row][lid]`，stride=16 保持合法
- **7.30 TFLOPS**（不及 K_TILE=16，SLM 大小影响占用率）

---

### 核心突破：预打包 fp16 输入（gemm_pack.cl）

**思路**：将 fp32 输入一次性预转换为 DPAS 友好的 fp16 布局，每次 benchmark 迭代直接读取 fp16 数据。

**好处**：
- 全局带宽减半（fp16 vs fp32）
- 内层循环消除 fp32→fp16 转换开销

**A_packed 布局**（`ushort`，共 M×K halves）：
```
offset = kt*(M*16) + m*16 + lid
```
kt = K-tile 索引（步长 16），m = 行号，lid = sub-group lane 0..15

**B_packed 布局**（`ushort`，共 K×N halves）：
```
offset = kt*(N/16)*256 + nt*256 + kk*16 + nn
```
每个 (kt, nt) tile 共 256 halves，内存连续

**打包内核**（`gemm_pack.cl`）：
- `pack_a`：gws=(16, M, K/16)，每 WI 打包 `A[m][kt*16+lid]`
- `pack_b`：gws=(16, 16, (K/16)*(N/16))，每 WI 打包 B 的一个 fp16 元素

**效果**：`18.33 TFLOPS`（**+2× vs 最优 SLM 版本**）

---

### 迭代 6：无 SLM + 预打包（单 sub-group per WG）
- lws=(16,1)，每 sg 输出 32M×64N tile = 4×4 float8 acc
- 每 K_TILE：4 A reads + 8 B reads + 16 DPAS，无 barrier
- **18.33 TFLOPS**（31.08% 峰值）

### 迭代 7：2×2 SLM + 预打包
- 4 sub-groups/WG，WG tile = 64M×128N
- SLM = 6 KB，barrier 少
- **20.21 TFLOPS**（34.27%）

### 迭代 8：4×2 SLM + 预打包，K_STEP=16
- 8 sub-groups/WG，WG tile = 128M×128N
- 每 sg：B 4× 重用，A 2× 重用
- **21.29 TFLOPS**（36.09%）

### 迭代 9：4×2 SLM + K_STEP=32（最优）
- 外循环步长 32K（两个 K_TILE=16），每次 barrier 处理 2 轮 DPAS
- SLM = 16 KB（两 K 条带）：A_slm 8 KB + B_slm 8 KB
- 每 sg 全局读：4+4=8 次 block_read_us8（A+B 各 2 条带）
- **26.35 TFLOPS**（**44.67% 峰值**）✅ **最优配置**

### 迭代 10：K_STEP=64
- SLM = 32 KB（占满），WG 占用率下降
- **22.37 TFLOPS**（回退，低占用率主导）

### 迭代 11：双缓冲 SLM（软件流水线）
- 尝试 load[next] 与 compute[cur] 重叠
- **22.28 TFLOPS**（无收益）

---

## 四、最优内核设计（gemm_kernel_dpas.cl）

```
编译选项: -cl-std=CL2.0 -cl-intel-256-GRF-per-thread

工作组:  lws = (16, 4, 2)   — 8 sub-groups/WG
全局组:  gws = (ceil(N/128)*16, ceil(M/128)*4, 2)

每 WG 输出 tile: 128M × 128N
每 sg  输出 tile:  32M ×  64N = 4 m-blocks × 4 n-blocks
累加器: 16 × float8 (128 个 fp32 寄存器/sg)

SLM: A_slm[2 条带][128 rows][16 K] = 8 KB
     B_slm[8 ntiles][2 条带][256 halves] = 8 KB
     合计 16 KB/WG（PTL 可同时 2 WG/Xe-core）

外层 K 循环（步长 32）:
  1. 8 sg 协作 global→SLM：每 sg 4+4 block_read_us8
  2. barrier
  3. 2 轮内层计算（每轮 16 K）：
       每 sg 读 4 A + 8 B SLM block_read_us8 + 16 DPAS
  4. barrier
```

---

## 五、性能汇总

### M=1024, K=4096, N=2048

| Kernel | avg (μs) | min (μs) | max (μs) | TFLOPS | %FP16 XMX | 加速比 |
|---|---:|---:|---:|---:|---:|---:|
| reference | 148,685 | 146,297 | 152,065 | 0.116 | 0.20% | 1.0× |
| kernel_foundry | 9,262 | 8,996 | 9,598 | 1.855 | 3.14% | 16.1× |
| copilot (DPAS) | **685** | **667** | 780 | **25.07** | **42.50%** | **216.9×** |

### M=2048, K=2048, N=2048

| Kernel | avg (μs) | min (μs) | max (μs) | TFLOPS | %FP16 XMX | 加速比 |
|---|---:|---:|---:|---:|---:|---:|
| reference | 103,873 | 87,030 | 125,297 | 0.165 | 0.28% | 1.0× |
| kernel_foundry | 8,473 | 8,424 | 8,578 | 2.027 | 3.44% | 12.3× |
| copilot (DPAS) | **653** | **630** | 717 | **26.30** | **44.59%** | **159.0×** |

### M=2048, K=2048, N=2560

| Kernel | avg (μs) | min (μs) | max (μs) | TFLOPS | %FP16 XMX | 加速比 |
|---|---:|---:|---:|---:|---:|---:|
| reference | 142,825 | 140,123 | 146,932 | 0.150 | 0.25% | 1.0× |
| kernel_foundry | 12,174 | 10,737 | 13,156 | 1.764 | 2.99% | 11.7× |
| copilot (DPAS) | **797** | **770** | 852 | **26.96** | **45.71%** | **179.3×** |

---

## 六、关键经验教训

### 布局规则
- `intel_sub_group_block_read_us8` 要求**行步长严格等于 sub-group 大小（16）**，SLM 和全局内存均适用
- DPAS B 操作数（`int8`）需将两个 `ushort8`（K=0..7 和 K=8..15）组合：
  ```c
  int8 b = as_int8((ushort16)(b0, b1));
  ```
- fp32→fp16 预打包应**一次完成**，在 benchmark 循环外执行

### GRF 模式
- 16 个 `float8` 累加器 = 256 GRF 行，超过小 GRF 模式（128 行/thread）上限
- **必须使用 `-cl-intel-256-GRF-per-thread`**，否则寄存器溢出严重拖慢性能
- 大 GRF 模式下每 EU thread 数从 10 降至 4，但 XMX 利用率大幅提升

### WG 结构选择
- **4×2 sub-groups/WG（8 sgs）** 是 PTL 上的最优配置：
  - 4× B 重用 + 2× A 重用
  - SLM 16 KB → 每 Xe-core 可同时放 2 个 WG
  - 比 4×4（16 sgs，barrier 开销大）和 2×2（4 sgs，重用不足）都更好

### SLM 与 barrier 权衡
- SLM 协作加载减少全局带宽，但每次全局→SLM 需两次 barrier
- **K_STEP=32（2 轮 DPAS per barrier pair）** 是最佳平衡点
  - K_STEP=16：barrier 太频繁
  - K_STEP=64：SLM 占满 32 KB，WG 占用率折半
- 双缓冲在此硬件上无收益（延迟已被 DPAS 计算基本覆盖）

### 当前瓶颈分析
达到 ~45% FP16 XMX 峰值后，继续提升需要：
- **`cl_intel_subgroup_2d_block_io`**：硬件加速的 2D block 读取 + VNNI 转置，可进一步减少 SLM 操作并消除数据重排开销
- 或更大的每 sg 输出 tile（需要更多 GRF）

---

## 七、文件清单

| 文件 | 说明 |
|---|---|
| `tests/cc/gemm_kernel_dpas.cl` | 最优 DPAS 内核（预打包 fp16 + 4×2 SLM + K_STEP=32） |
| `tests/cc/gemm_pack.cl` | fp32→fp16 预打包内核（pack_a / pack_b） |
| `tests/cc/gemm_kernel_kf.cl` | kernel_foundry fp32 SLM-tiled 对比内核 |
| `tests/cc/gemm_reference.cl` | 朴素 fp32 参考内核（每 WI 一个输出元素） |
| `tests/cc/gemm_test.cpp` | C++ 测试框架（正确性 + 三核对比 benchmark） |
| `tests/cc/build.bat` | Windows 编译脚本（Intel oneAPI icx） |
| `tests/cc/run.bat` | cliloader 性能采集脚本 |
