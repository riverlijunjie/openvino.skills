# CLILoader Performance Analysis — perf_dGPU.log

## Hardware & Environment
- **Device**: Intel(R) Arc(TM) Pro B60 Graphics (160 CUs, 2400 MHz)
- **Est. Peak FP16**: ~12.3 TFLOPS
- **Est. Memory BW**: ~560 GB/s (GDDR6)
- **Total GPU Time**: 24,799 ms (24.8s)
- **Total Enqueues**: 64,173
- **Workload**: LLM inference (MoE model, likely DeepSeek-like), prefill + decode

---

## 1. Category Summary (按时间排序)

| Category       | Total (ms) |  Time % | Calls  | Avg (us) | Notes                       |
|:---------------|----------:|--------:|-------:|---------:|:----------------------------|
| **GEMM**       |  16,943.6 |  68.32% | 20,144 |    841   | 绝对主导，所有matmul计算        |
| **HtoD Copy**  |   4,551.3 |  18.35% |    776 |  5,865k  | 模型权重/KV-cache搬运          |
| **MOE**        |   2,974.2 |  11.99% | 16,336 |    182   | MoE专用kernel (scatter/gather等) |
| PagedAttn      |      96.7 |   0.39% |  1,488 |     65   | decode阶段attention            |
| SDPA (prefill) |      78.2 |   0.32% |     48 |  1,629   | prefill阶段attention           |
| RMS            |      47.6 |   0.19% |  6,176 |      8   | RMSNorm                       |
| MtoH Copy      |      23.2 |   0.09% | 10,336 |      2   | 小块readback (metadata)        |
| ROPE           |      20.1 |   0.08% |  3,072 |      7   | Rotary Position Embedding     |
| PA Finalization|       9.5 |   0.04% |  1,488 |      6   | PA reduction step             |
| Reorder        |       9.0 |   0.04% |    480 |     19   | 布局转换                       |
| DynQuant       |       6.8 |   0.03% |    144 |     47   | 动态量化                       |
| DtoH Copy      |       5.4 |   0.02% |     32 |    169   | 输出搬回host                   |
| MemFill        |       1.7 |   0.01% |    340 |      5   | 清零                          |

**关键发现**: GEMM (68%) + HtoD (18%) + MOE (12%) = **98.66%** 的总时间。优化必须聚焦在这三个类别。

---

## 2. GEMM Kernel 详细分析

GEMM共 16,943.6 ms (68.32%)，是性能的绝对瓶颈。按dispatch配置分组（Top 19，>0.3%）：

| # | Total(ms) | Time% | Calls | Avg(us) | Max/Min | GWS                  | Config        |
|---|----------:|------:|------:|--------:|--------:|:---------------------|:--------------|
| 1 |   2,325.8 | 9.38% | 1,642 | 1,416.5 |    3.1x | 64×48×2              | REG128 SLM=16K |
| 2 |   1,425.0 | 5.75% |   498 | 2,861.4 |    3.9x | 2560×4×1             | REG256        |
| 3 |   1,395.3 | 5.63% |   310 | 4,501.0 |    2.6x | 384×24×1             | REG256        |
| 4 |   1,344.5 | 5.42% |   821 | 1,637.6 |    3.0x | 64×128×2             | REG128 SLM=16K |
| 5 |   1,216.9 | 4.91% |   444 | 2,740.8 |    1.6x | 2304×4×1             | REG256        |
| 6 |   1,116.0 | 4.50% |   814 | 1,371.0 |    1.8x | 384×12×1             | REG256        |
| 7 |   1,083.9 | 4.37% |   336 | 3,225.8 |    2.1x | 384×20×1             | REG256        |
| 8 |     735.7 | 2.97% |    98 | 7,507.6 |    1.6x | 2560×4×1 (LWS128)   | REG256        |
| 9 |     706.3 | 2.85% |   358 | 1,972.8 |    1.5x | 384×16×1             | REG256        |
|10 |     637.9 | 2.57% |   138 | 4,622.3 |    1.6x | 1280×8×1             | REG256        |
|11 |     629.0 | 2.54% |   670 |   938.9 |    1.7x | 384×8×1              | REG256        |
|12 |     601.9 | 2.43% |   484 | 1,243.6 |    1.8x | 1920×4×1             | REG256        |
|13 |     552.1 | 2.23% |   496 | 1,113.1 |    1.7x | 32×48×2              | REG128 SLM=16K |
|14 |     523.8 | 2.11% |   335 | 1,563.6 |    1.5x | 1024×8×1             | REG256        |
|15 |     513.1 | 2.07% |   247 | 2,077.3 |    1.3x | 1024×10×1            | REG256        |
|16 |     440.9 | 1.78% |   464 |   950.3 |    1.9x | 1024×6×1             | REG256        |
|17 |     323.1 | 1.30% |   636 |   508.0 |    2.2x | 32×48×1              | REG128        |
|18 |     268.7 | 1.08% |   248 | 1,083.6 |    1.9x | 32×128×2             | REG128 SLM=16K |
|19 |     236.9 | 0.96% |   318 |   745.1 |    2.2x | 32×128×1             | REG128        |

### GEMM 观察

1. **两种主要GEMM模式**：
   - **SLM=16384 + REG128** (带共享内存tiling): #1, #4, #13, #18 → 合计 ~5,491ms (22.1%)
     - 这些是 prefill 阶段的大矩阵乘法（batch=2的z维度暗示2-way batch split）
   - **REG256 + 无SLM** (寄存器密集型): #2,#3,#5-#12,#14-#16 → 合计 ~10,006ms (40.4%)
     - GWS z=1 说明是 decode 阶段的 GEMV（矩阵-向量乘）

2. **Max/Min ratio 观察**: 
   - #1 和 #2 的 Max/Min 比值 >3x，说明存在**显著的执行时间波动**，可能是线程争用或L3 cache thrashing
   - #2 (GWS 2560×4×1) 的 Max/Min=3.9x 最严重

3. **Decode GEMM 效率估算**：
   - 以 #2 为例：GWS=2560×4×1，LWS=32×4×1，SIMD16 REG256
   - 每次调用 avg=2.86ms，对于 decode 阶段的 weight-only 量化 GEMV 来说，这个时间较长
   - 暗示可能存在内存带宽瓶颈（decode是memory-bound的）

---

## 3. HtoD 传输分析

总计 4,551.3 ms (18.35%)，**所有传输带宽一致约 3.3 GB/s**。

| Block Size | × Calls | Total (GB) | Time (ms) | BW (GB/s) |
|------------|---------|------------|-----------|-----------|
| 100.7 MB   | 144     | 14.50      | 4,085.0   | 3.3       |
| 311.2 MB   | 2       | 0.62       | 175.3     | 3.3       |
| 3.1 MB     | 144     | 0.45       | 127.9     | 3.3       |
| 5.2 MB     | 48      | 0.25       | 71.1      | 3.3       |
| 4.2 MB     | 48      | 0.20       | 56.9      | 3.3       |
| 0.8 MB     | 144     | 0.11       | 32.3      | 3.3       |

### HtoD 观察

- **3.3 GB/s 远低于 PCIe 4.0 x16 理论峰值 (~32 GB/s)**！
  - 这可能意味着数据走的是 **PCIe x4 或 x8** 通道，或者存在PCIe bandwidth限制
  - 另一种可能：host内存分配方式（non-pinned memory）导致需要额外拷贝
- 100.7 MB × 144 次 = 14.5 GB，占 HtoD 的 89.8%
  - 这是模型权重的逐层加载。144次 = 48层 × 3个矩阵（Q/K/V或gate/up/down）
  - 每层100.7MB → 整个模型约 14.5GB，符合一个中等规模LLM
- **311.2 MB × 2** 可能是 embedding 或 lm_head 权重

### 优化建议
- 检查PCIe链路是否为x16全速
- 若权重可以常驻GPU显存，减少重复HtoD传输（在多次推理间复用）

---

## 4. MOE Kernel 分析

总计 2,974.2 ms (11.99%)，这是一个**MoE (Mixture-of-Experts) 模型**。

### MOE 子kernel类型分解

| Sub-kernel        | Total (ms) | Calls | Dispatch Configs | Notes                      |
|:------------------|----------:|------:|-----------------:|:---------------------------|
| fuse_gather       |   1,168.5 | 5,168 |              753 | 从路由结果gather expert数据  |
| fuse_index_add    |   1,159.7 | 5,168 |              753 | scatter-add结果回去          |
| mlp_down          |     409.1 | 1,488 |                1 | Expert MLP down projection  |
| mlp_gate_up       |     183.3 | 1,488 |                1 | Expert MLP gate+up projection|
| fuse_softmax_topk |      31.4 | 1,536 |                2 | Router softmax + top-k选择  |
| mlp_reduce        |      22.1 | 1,488 |                1 | Expert输出reduce             |

### MOE 观察

1. **fuse_gather + fuse_index_add 占 MOE 的 78.3%**（2,328 ms）：
   - 这两个是数据shuffle操作（gather input → experts, scatter output → tokens）
   - 753个不同的dispatch配置说明GWS随每次推理的token数变化（dynamic shaped）
   - 这是 MOE 推理中典型的**数据搬运瓶颈**，不是计算瓶颈

2. **mlp_down (409ms) 远大于 mlp_gate_up (183ms)**：
   - down_proj 的隐藏维度通常是 gate_up 输出维度的 2x（因为 SwiGLU 的 gate 和 up 合并），
     但这里时间比为 2.2x，基本合理
   - GWS=[8×32×512] vs [8×32×192] 体现了不同的输出维度

---

## 5. Attention 分析

总计 184.4 ms (0.74%)，比较健康。

| Kernel          | Total (ms) | Calls | Avg (us) | Notes                    |
|:----------------|----------:|------:|---------:|:-------------------------|
| PA decode       |      96.7 | 1,488 |     65.0 | Single-token paged attention |
| SDPA prefill    |      78.2 |    48 |  1,629.3 | Prefill阶段的micro SDPA     |
| PA finalization |       9.5 | 1,488 |      6.4 | PA reduction                |

- **1488 次 PA_decode** = 48 层 × 31 个 decode token（即 prompt 后生成了 31 个 token）
- **48 次 SDPA_prefill** = 48 层 × 1 次 prefill
- PA decode 平均 65us/call，对于 Arc B60 来说是合理的

---

## 6. 其他 Kernel

| Kernel     | Total (ms) |  Time % | Calls | Avg (us) |
|:-----------|----------:|--------:|------:|---------:|
| RMS        |     47.6  |  0.19%  | 6,176 |      7.7 |
| MtoH Copy  |     23.2  |  0.09%  |10,336 |      2.2 |
| ROPE       |     20.1  |  0.08%  | 3,072 |      6.5 |
| Reorder    |      9.0  |  0.04%  |   480 |     18.8 |
| DynQuant   |      6.8  |  0.03%  |   144 |     47.2 |
| DtoH Copy  |      5.4  |  0.02%  |    32 |    168.9 |
| MemFill    |      1.7  |  0.01%  |   340 |      5.1 |

这些小kernel加在一起 < 0.5%，不需要优化。

---

## 7. 关键发现总结

### 🔴 问题 1：HtoD 带宽仅 3.3 GB/s（理论 ~32 GB/s）
- **严重度**: HIGH
- **影响**: 白白浪费 4.5 秒在数据传输上（18% 的总时间）
- **根因推测**: PCIe 链路未跑满，可能是 x4/x8 连接，或 host 端使用了 non-pinned memory
- **效率**: 3.3 / 32 = **10.3% PCIe4 x16 利用率**

### 🟡 问题 2：MOE gather/scatter 占 2.3 秒
- **严重度**: MEDIUM
- **影响**: 12% 的总时间花在数据路由上，不是计算
- **根因**: fuse_gather + fuse_index_add 是 memory-bound 操作，753 种 dispatch 配置说明每次调用的 shape 不同
- **优化方向**: 1) 考虑 fuse 更多操作减少 kernel launch overhead 2) 优化 gather/scatter 的内存访问模式

### 🟡 问题 3：Top GEMM kernel Max/Min 波动 >3x
- **严重度**: MEDIUM  
- **影响**: #1 GEMM (9.4%) 的执行时间波动从 790us 到 2437us，3.1x 变化
- **根因推测**: L3 cache contention, EU occupancy 波动, 或其他 kernel 的 SLM 争抢
- **优化方向**: 分析是否存在 kernel 执行顺序依赖导致的 cache 冷热差异

### 🟢 正面发现：Attention 效率良好
- PA decode 65us/call, SDPA prefill 1.6ms/call 都在合理范围
- Attention 仅占 0.74%，模型的主要计算已经很好地被 GEMM 吸收

---

## 8. 模型结构推断

| 特征               | 推断值                              |
|:-------------------|:------------------------------------|
| 模型类型           | MoE LLM (类 DeepSeek / Mixtral)      |
| 层数               | 48 (from SDPA prefill calls)         |
| 生成token数        | 31 (from PA decode = 1488/48)        |
| 模型大小（权重）   | ~14.5 GB (from HtoD 100.7MB × 144)   |
| MoE Expert数       | 8 (from GWS first dim = 8)           |
| Hidden dim         | 2048 (from GWS patterns)             |
| Head数             | 32 (from PA GWS y=32)               |

---

## 9. MOE_USE_MICRO_GEMM_PREFILL=1 性能分析

### 9.1 环境与关键对比

| 参数 | 值 |
|:-----|:---|
| 开启功能 | `MOE_USE_MICRO_GEMM_PREFILL = 1` |
| 模型 | Qwen3-30B-A3B (128 experts, top_k=8) |
| hidden_size / inter_size | 2048 / 768 (per expert) |
| 权重格式 | INT4, group_size=128 |
| seq_len | 2048 |

| 指标 | BMG dGPU (Arc B60) | PTL iGPU (12 Xe Cores) |
|:-----|:-------------------|:----------------------|
| Xe Cores | 20 | 12 |
| Memory BW | 560 GB/s GDDR6 (VRAM) | ~90 GB/s LPDDR5 (shared) |
| FP16 compute | ~12.3 TFLOPS | ~3.7 TFLOPS |
| Prefill time | **~70s (MOE 68.4s)** | **~1s** |
| 预期 (BMG应更快) | **<1s** | ~1-2s |

**核心矛盾**: BMG 拥有更强算力 (3.3x) 和更高带宽 (6.2x)，prefill 却慢了 **70x**。

### 9.2 Micro GEMM Kernel 时间分布

| Kernel | Calls | Time (s) | % | Avg (ms) |
|:-------|------:|--------:|----:|--------:|
| moe_gather_ref_prefill | 48 | 28.52 | 38.45% | 594.2 |
| moe_gemm_prefill_down | 48 | 21.78 | 29.36% | 453.7 |
| moe_gemm_prefill_gate | 48 | 9.21 | 12.41% | 191.8 |
| moe_gemm_prefill_up | 48 | 8.87 | 11.96% | 184.8 |
| HtoD transfer | 776 | 4.55 | 6.13% | 5.9 |
| 其他 | — | 1.25 | 1.69% | — |
| **MOE 合计** | | **68.4** | **92.2%** | |

### 9.3 根因: INTERNAL_BUFFER 分配在 usm_host（系统内存）

#### 9.3.1 问题本质: GPU 通过 PCIe 访问系统内存

verbose_micro.log 中 `set_kernel_arg` 日志直接暴露了问题：

```
moe_gemm_gate kernel arguments:
  arg 0: (usm_host) 67 MB  ← scratch.x (GEMM 输入) 在系统内存!
  arg 1: (usm_device) 96 MB ← weights 在 VRAM ✓
  arg 2: (usm_host) 25 MB  ← scratch.up (silu_mul 输入) 在系统内存!
  arg 3: (usm_host) 25 MB  ← scratch.gate (GEMM 输出) 在系统内存!
  arg 9: (usm_device) 3 MB  ← scale weights 在 VRAM ✓
  arg 10: (usm_device) 768 KB ← zp weights 在 VRAM ✓
```

MOE 4 个大型中间缓冲区全部在 `usm_host`（系统 RAM），而非 `usm_device`（VRAM）：

| Buffer | 用途 | 大小 | 当前分配 | GPU 访问路径 |
|:-------|:-----|-----:|:---------|:------------|
| Buffer 4 (scratch.x) | gather 输出 / GEMM 输入 | 67 MB | **usm_host** | PCIe 3.3 GB/s |
| Buffer 2 (scratch.up) | up GEMM 输出 | 25 MB | **usm_host** | PCIe 3.3 GB/s |
| Buffer 6 (scratch.gate) | gate GEMM 输出 | 25 MB | **usm_host** | PCIe 3.3 GB/s |
| Buffer 3 (scratch.y) | down GEMM 输出 | 67 MB | **usm_host** | PCIe 3.3 GB/s |
| **合计** | | **184 MB** | | |

#### 9.3.2 为什么 iGPU 不受影响

- **iGPU (PTL)**: `usm_host` 和 `usm_device` 是同一块物理内存 (LPDDR5)，GPU 以 ~90 GB/s 访问，**零惩罚**
- **dGPU (BMG)**: `usm_host` = 系统 RAM，GPU 必须通过 PCIe 总线访问，带宽从 **560 GB/s → 3.3 GB/s**（170x 降级）

#### 9.3.3 代码追踪

**Step 1**: `moe_3gemm_swiglu_opt.cpp` 中 `get_internal_buffer_descs()` 为所有 buffer 设置 `lockable=true`:

```cpp
// moe_3gemm_swiglu_opt.cpp:1035-1044
internal_buffers.emplace_back(layout_gateup_out, true);  // 2: up output    ← lockable=TRUE
internal_buffers.emplace_back(layout_down_out, true);    // 3: down output   ← lockable=TRUE
internal_buffers.emplace_back(layout_down_out, true);    // 4: gate/up input ← lockable=TRUE
internal_buffers.emplace_back(layout_gateup_out, true);  // 6: gate output   ← lockable=TRUE
```

**Step 2**: `primitive_inst.cpp:2417-2425` 中 `allocate_internal_buffer()` 的分配逻辑:

```cpp
if (available_device_mem >= layout.bytes_count()
    && (input_device_mem || onednn)
    && !lockable) {                          // ← lockable=true 时此条件永远为 false!
    alloc_type = get_preferred_memory_allocation_type();    // usm_device
} else {
    alloc_type = get_lockable_preferred_memory_allocation_type(); // usm_host!
}
```

**`lockable=true` 强制所有 INTERNAL_BUFFER 分配为 `usm_host`**，无论 device memory 是否充足。

#### 9.3.4 这些 buffer 是否真需要 CPU 访问？

追踪完整 execution flow，buffers 2/3/4/6 **仅被 GPU kernel 访问**：
- Buffer 4: gather GPU kernel 写入 → up/gate GEMM GPU kernel 读取
- Buffer 2: up GEMM GPU kernel 写入 → gate GEMM GPU kernel 读取 (silu_mul)
- Buffer 6: gate GEMM GPU kernel 写入 → down GEMM GPU kernel 读取
- Buffer 3: down GEMM GPU kernel 写入 → scatter GPU kernel 读取

**没有任何 CPU 操作需要这些 buffer 是 lockable 的。**

### 9.4 PCIe 带宽损失量化

#### 每层 PCIe 流量

| 阶段 | 从 host 读 | 写入 host | PCIe 总流量 |
|:-----|----------:|--------:|----------:|
| Gather | 0.07 MB | 67.1 MB | 67.2 MB |
| Up GEMM | 67.3 MB | 25.2 MB | 92.5 MB |
| Gate GEMM | 92.5 MB | 25.2 MB | 117.7 MB |
| Down GEMM | 25.4 MB | 67.1 MB | 92.5 MB |
| Scatter | 67.1 MB | — | 67.1 MB |
| **合计/layer** | **252 MB** | **185 MB** | **437 MB** |
| **合计/48 layers** | | | **21.0 GB** |

#### 有效 PCIe 带宽 vs 理论

| Kernel | 耗时 (ms) | PCIe 流量 (MB) | 有效 BW (GB/s) |
|:-------|--------:|------------:|-------------:|
| Gather | 594.2 | 67.2 | 0.11 |
| Up GEMM | 184.8 | 92.5 | 0.50 |
| Gate GEMM | 191.8 | 117.7 | 0.61 |
| Down GEMM | 453.7 | 92.5 | 0.20 |
| **平均** | | | **0.31** |

- 理论 PCIe 4.0 x4: 3.94 GB/s
- 实际有效: **0.31 GB/s** (仅 **7.9%** 利用率)
- GPU 发起的 host memory 随机访问模式导致 PCIe 利用率极低

### 9.5 修复后预期性能 (usm_device)

修改 buffers 2/3/4/6 为 `lockable=false` 后，分配将转为 `usm_device` (VRAM 560 GB/s):

| 阶段 | Memory-bound (ms) | Compute-bound (ms) | 瓶颈 | 预期耗时 (ms) |
|:-----|:-------:|:----------:|:----:|:--------:|
| Gather | 0.13 | N/A | Memory | 0.13 |
| Up GEMM | 0.28 | **4.19** | Compute | 4.19 |
| Gate GEMM | 0.33 | **4.19** | Compute | 4.19 |
| Down GEMM | 0.28 | **4.19** | Compute | 4.19 |
| Scatter | 0.13 | N/A | Memory | 0.13 |
| **合计/layer** | | | | **12.8** |

| 指标 | 当前 (usm_host) | 修复后 (usm_device) | 变化 |
|:-----|:-----------:|:---------------:|:----:|
| 每层 MOE | 1424 ms | 12.8 ms | **111x** |
| 48 层 MOE | 68.4s | **0.62s** | **111x** |
| First token latency | ~70s | **<2s** | **35x+** |

#### 与 PTL iGPU 对比

| 平台 | Xe Cores | 每层 MOE (ms) | 48 层 (s) | 状态 |
|:-----|:--------:|:--------:|:------:|:-----|
| PTL iGPU (实际) | 12 | ~21 | ~1.0 | ✓ 正常 |
| BMG dGPU (当前) | 20 | **1424** | **68.4** | usm_host 灾难 |
| BMG dGPU (修复后) | 20 | **12.8** | **0.62** | compute-bound ✓ |

修复后 BMG 应比 PTL **快 ~1.6-3.4x**（更多 Xe Cores + 更高 VRAM 带宽）。

### 9.6 修复方案与对 iGPU 的影响

#### 代码修改

文件: `src/plugins/intel_gpu/src/graph/impls/ocl_v2/moe/moe_3gemm_swiglu_opt.cpp`

```cpp
// 修改前 (全部 lockable=true):
internal_buffers.emplace_back(layout_gateup_out, true);  // 2: up output
internal_buffers.emplace_back(layout_down_out, true);    // 3: down output
internal_buffers.emplace_back(layout_down_out, true);    // 4: up/gate input
internal_buffers.emplace_back(layout_gateup_out, true);  // 6: gate output

// 修改后 (GPU-only buffers 设为 lockable=false):
internal_buffers.emplace_back(layout_gateup_out, false);  // 2: up output    (GPU-only)
internal_buffers.emplace_back(layout_down_out, false);     // 3: down output  (GPU-only)
internal_buffers.emplace_back(layout_down_out, false);     // 4: up/gate input (GPU-only)
internal_buffers.emplace_back(layout_gateup_out, false);   // 6: gate output  (GPU-only)
```

保持 `lockable=true` 的 buffer:
- Buffer 0/1 (topk_id/weights): CPU `copy_to` 读取路由结果
- Buffer 5 (routing_weights): 小尺寸 (32KB)
- Buffer 7/8 (expert_masks): CPU `copy_from` 写入路由信息
- Buffer 9-13 (micro routing): CPU 读写

#### 对 iGPU 的影响分析

**结论: 不影响 iGPU 性能**

1. `lockable=false` 不等于 "一定分配 usm_device"。分配逻辑还需要 `input_device_mem=true` 条件才会选 `usm_device`
2. **iGPU 上 usm_host = usm_device = 同一块物理内存** (LPDDR5 ~90 GB/s)。无论分配类型如何改变，GPU 访问带宽完全相同
3. `usm_device` 上的 `copy_from()`/`copy_to()` 使用 DMA (`clEnqueueMemcpy`) 传输，仍正常工作
4. `gpu_usm::lock()` 对 `usm_device` 自动创建 host shadow buffer 并 memcpy（只支持 read），功能不受影响

| 属性 | iGPU | dGPU |
|:-----|:-----|:-----|
| usm_host 物理位置 | 系统内存 (LPDDR5) | 系统内存 (DDR) |
| usm_device 物理位置 | **同一块系统内存** | VRAM (GDDR6) |
| GPU 访问 usm_host 带宽 | ~90 GB/s | **3.3 GB/s (PCIe)** |
| GPU 访问 usm_device 带宽 | ~90 GB/s | 560 GB/s |
| 修改后影响 | **零影响** | **111x 加速** |

#### 全部 Buffer 改为 lockable=false（usm_device）的影响评估

如果将全部 14 个 internal buffer 都从 `lockable=true` 改为 `lockable=false`，各 buffer 影响如下：

| Buffer | 大小 | CPU 操作 | 改为 false 后影响 |
|:-------|:-----|:---------|:----------------|
| 0 (topk_id) | 64KB | `copy_to` 读取路由 | 多 1 次 DMA 拷贝 64KB，可忽略 |
| 1 (topk_weights) | 32KB | `copy_to` 读取路由 | 多 1 次 DMA 拷贝 32KB，可忽略 |
| **2 (up out)** | **25MB** | **无 CPU 操作** | **✓ 应该改** |
| **3 (down out)** | **67MB** | **无 CPU 操作** | **✓ 应该改** |
| **4 (gather in)** | **67MB** | **无 CPU 操作** | **✓ 应该改** |
| 5 (routing) | 32KB | 间接使用 | 太小，无所谓 |
| **6 (gate out)** | **25MB** | **无 CPU 操作** | **✓ 应该改** |
| 7 (mask_batch) | 1MB | `copy_from` 写入 | 多 1 次 DMA，可接受 |
| 8 (mask_topk) | 1MB | `copy_from` 写入 | 多 1 次 DMA，可接受 |
| 9-11 (micro routing) | 1MB × 3 | `copy_from` 写入 | 多 DMA，可接受 |
| 12 (token_idx) | 64KB | `copy_from` 写入 | 可忽略 |
| 13 (expert_num) | 4B | `mem_lock` 读取 | ⚠️ 有 lock，但只 4 字节 |

**结论**：
- **必须改的 4 个** (buf 2/3/4/6)：大尺寸 GPU-only buffer，合计 184MB，是性能瓶颈根因
- **改了无害的 8 个** (buf 0/1/5/7/8/9-12)：小尺寸，DMA 开销可忽略
- **需注意的 1 个** (buf 13)：使用 `mem_lock` 读取，改为 `usm_device` 后 lock 会创建临时 host buffer + memcpy，但仅 4 字节，开销为零
- **全部改为 false 是安全的**，但最小改动方案只需改 buf 2/3/4/6
