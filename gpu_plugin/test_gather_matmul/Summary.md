# GatherMatmul 深度分析与先前 MoE 实现对比

> 本文档基于对 OpenVINO GPU Plugin 源码的逐文件分析，记录 `GatherMatmul` 算子的完整实现逻辑，并与旧版 MoE 实现做精准的代码级对比与性能分析。

---

## 一、 背景：两套 MoE 范式

OpenVINO GPU Plugin 中存在两套 MoE 实现范式：

### 旧方案（`moe_gemm` 体系）— 分离的"两套班子"

针对 **单 Token（Decode）** 和 **多 Token（Prefill）** 采用了完全独立的代码路径：

| 阶段 | Kernel 体系 | 核心思路 |
|:---|:---|:---|
| **Decode（单 Token）** | `moe_3gemm_swiglu_fuse.cl` | 单个巨型融合 Kernel，Gate/Up/SwiGLU/Down 全在寄存器内完成，零中间显存 I/O |
| **Prefill（多 Token）** | `moe_mask_gen` → `moe_gather_ref` → `moe_gemm` → `moe_scatter_reduction` | 散装 Kernel 组合，利用 GPU 并行前缀和完成 Token 分组，Grouped GEMM 共享权重 |

### 新方案（`GatherMatmul` 体系）— 统一的"乐高积木"

用一个通用算子 `GatherMatmul` 统一表达"查询索引 + 批量矩阵乘"，在底层按 `n_tokens` 大小自动切换执行路径：

```
n_tokens == 1       → regular_micro_single_token  (gather_matmul.cl)
1 < n_tokens ≤ 16  → regular_micro_multi_tokens   (gather_matmul.cl, 每 token 单独 dispatch)
n_tokens > 16       → batched 三阶段流水线          (Sort → Gather → gather_matmul_batched.cl)
```

---

## 二、 GatherMatmul 完整执行流程

### 2.1 C++ 侧决策逻辑（`gather_matmul.cpp`）

```
GatherMatmulOCLImpl::execute()
     │
     ├─ is_prefill_stage()? → n_tokens > 1?
     │       │
     │       ├─ Yes (Prefill)
     │       │       └─ use_batched_prefill()? → n_tokens > BATCHED_PREFILL_THRESHOLD(16)?
     │       │               ├─ Yes → [Sort] → [Gather] → [batched_gemm] 三阶段流水线
     │       │               └─ No  → regular_micro_multi_tokens (per-token dispatch)
     │       │
     │       └─ No (Decode, n_tokens == 1)
     │               └─ regular_micro_single_token
```

### 2.2 七个内部缓冲区（中转站）

批量路径中，各 Kernel 通过以下内部缓冲区传递数据：

| 编号 | 名称 | 类型 | 大小 | 语义 |
|:---:|:---|:---:|:---|:---|
| 0 | `GATHERED_A` | f16 | `n_tokens × top_k × K` | 按 Expert 组重排后的连续 Activation |
| 1 | `GROUP_EXPERT_IDS` | i32 | `max_groups` | 每组对应的 Expert ID |
| 2 | `GROUP_SLOT_IDS` | i32 | `max_groups` | 每组对应 top_k 里的第几个 Slot |
| 3 | `GROUP_OFFSETS` | i32 | `max_groups` | 每组在 `GATHERED_A` 中的起始偏移（行数） |
| 4 | `GROUP_SIZES` | i32 | `max_groups` | 每组包含的 Token 数量 |
| 5 | `TOKEN_MAP` | i32 | `max(n_tokens×top_k, max_groups)` | 排序后位置 → 原始 Token 下标的映射 |
| 6 | `NUM_GROUPS` | i32 | 1 | 实际活跃的 Group 数量 |

> `max_groups = n_all_experts × top_k`，即每个（Expert, Slot）组合对应一个潜在 Group。

---

## 三、 各 Kernel 精准分析

### Kernel A：`gather_matmul.cl` → `batch_gather_matmul`（Per-token 路径）

**适用场景**：`n_tokens ≤ 16` 或 Decode 阶段（`n_tokens == 1`）

#### 输入
| 参数 | 形状 | 说明 |
|:---|:---|:---|
| `input_ptr` (A) | `[n_act, n_tokens, K]` | Token Activations。第一个 GatherMatmul 时 `n_act=1`（广播），后续 `n_act=top_k` |
| `weight_ptr` (B) | `[n_all_experts, N, K]` | 所有专家权重（K 维已转置） |
| `indices` | `[n_tokens, top_k]` | 每个 Token 的专家路由索引 |
| `m`, `k` | scalar | 输出特征维度 N 和 reduction 维度 K |

#### Dispatch 策略
```
global[0] = ceil(M / wg_tile_m)    ← 输出特征方向 tile 数
global[1] = 1                       ← 每次处理1个 Token
global[2] = n_tokens × top_k       ← 所有 (token, slot) 对的展开
```

从 `flat_idx = get_group_id(2)` 解码：
```c
token_idx   = flat_idx / top_k;
expert_slot = flat_idx % top_k;
expert_id   = indices[token_idx * top_k + expert_slot];  // 查路由表
```

#### 核心计算
```c
input_ptr  += a_slot * n_tokens * K + token_idx * K;     // 当前 token 行
weight_ptr += expert_id * EXPERT_STRIDE;                   // 当前 expert 权重
out_ptr    += expert_slot * n_tokens * M + token_idx * M; // 输出位置

// cur_n_tokens=1，本质是 GEMV
c_tile = ugemm_gm(weight_ptr, input_ptr, M, 1, K, ...);
tile_store(c_tile, out_ptr);
```

#### 输出
| 形状 | 说明 |
|:---|:---|
| `[top_k, n_tokens, N]` | 每个 `(slot, token)` 的 MLP 投影结果，顺序写入 |

#### 关键问题
若多个 Token 路由到同一个 Expert，每个 Workgroup 会**独立地重复加载**该 Expert 的权重，造成显存带宽浪费（Memory Bandwidth Thrashing）。

---

### Kernel B1：`gathermatmul_sort.cl` → `bgm_sort`（Sort 阶段）

**适用场景**：`n_tokens > 16` 的 Batched 路径

#### 输入
| 参数 | 形状 | 说明 |
|:---|:---|:---|
| `indices` (INPUT0) | `[n_tokens, top_k]` | 每个 Token 的专家路由索引 |

#### Dispatch
```
global = {1, 1, 1}，local = {1, 1, 1}   ← 单线程，纯顺序执行
```

#### 执行逻辑（3遍扫描）

**Pass 1 — 直方图计数**：用 `token_map` 作为 scratch 统计每个 `(slot, expert)` bin 有多少 Token。

**Pass 2 — 压缩 + 前缀和**：跳过空 bin，为非空 bin 分配紧凑的 Group ID，计算 `group_offsets`（prefix sum）。

**Pass 3 — Token 散射**：对每个 Group，遍历所有 Token，找出属于该 `(slot, expert)` 的 Token，写入 `token_map`。

#### 输出（写入内部缓冲区）
`GROUP_EXPERT_IDS`, `GROUP_SLOT_IDS`, `GROUP_OFFSETS`, `GROUP_SIZES`, `TOKEN_MAP`, `NUM_GROUPS`

#### 关键问题
**全程串行**，无法利用 GPU 的大量并行计算单元（对比旧方案 `moe_mask_gen` 的 GPU 并行前缀扫描）。

---

### Kernel B2：`gathermatmul_gather.cl` → `bgm_gather`（Gather 阶段）

#### 输入
| 参数 | 形状 | 说明 |
|:---|:---|:---|
| `input_ptr` (A) | `[n_act, n_tokens, K]` | 原始 Token Activations |
| `token_map` | `[n_tokens × top_k]` | Sort 输出的位置-Token 映射 |
| `group_slot_ids` | `[max_groups]` | 每组的 Slot 索引 |
| `group_offsets` | `[max_groups]` | 每组在 `GATHERED_A` 中的起始偏移 |
| `group_sizes` | `[max_groups]` | 每组 Token 数量 |
| `num_groups` | `[1]` | 活跃组数 |

#### Dispatch（3D 高度并行）
```
global[2] = max_groups                    ← 每组一个 z-slice
global[1] = max_tokens_per_group          ← 每个 Token 的 y-dim
global[0] = ceil(K / COPY_BLOCK=8)       ← K 维并行拷贝
```

#### 核心计算（128-bit 向量化拷贝）
```c
int orig_token = token_map[group_offsets[group_id] + token_in_group];
int a_slot     = min(slot, n_act - 1);   // 广播截断

// 源：A[a_slot, orig_token, :] — 随机访问
src = input_ptr + (a_slot * n_tokens + orig_token) * K;
// 目标：GATHERED_A[offset + token_in_group, :] — 连续写
dst = gathered_ptr + (group_offsets[group_id] + token_in_group) * K;

half8 val = vload8(0, src + k_offset);  // 128-bit 向量化读
vstore8(val, 0, dst + k_offset);        // 128-bit 向量化写
```

#### 输出
| 缓冲区 | 形状 | 内容 |
|:---|:---|:---|
| `GATHERED_A` | `[n_tokens × top_k, K]` | 按 Group 排列的连续 Activation。Group 0 的所有 Token 行紧挨着，然后是 Group 1 的，以此类推 |

---

### Kernel B3：`gather_matmul_batched.cl` → `gather_matmul_batched`（Batched GEMM 阶段）

#### 输入
| 参数 | 形状 | 说明 |
|:---|:---|:---|
| `gathered_input_ptr` | `[n_tokens×top_k, K]` | Gather 输出的连续 Activation |
| `weight_ptr` (B) | `[n_all_experts, N, K]` | 所有专家权重（转置） |
| `group_expert_ids` | `[max_groups]` | 每组用哪个 Expert 的权重 |
| `group_slot_ids` | `[max_groups]` | 每组属于哪个 Slot |
| `group_offsets` | `[max_groups]` | 每组在连续缓冲区的起始行 |
| `group_sizes` | `[max_groups]` | 每组 Token 数量 |
| `token_map` | `[n_tokens×top_k]` | 散射写回时恢复原始 Token 位置 |
| `num_groups` | `[1]` | 活跃 Group 数 |

#### Dispatch
```
global[0] = ceil(M / wg_tile_m)          ← 输出特征 tile
global[1] = ceil(n_tokens / wg_tile_n)   ← Token tile（按 group_size 早退）
global[2] = max_groups                    ← 一个 Group 一个 z 切片
```

#### 核心计算（两步走）

**Step 1 — Batched GEMM（一次权重加载服务多 Token）**：
```c
int expert_id    = group_expert_ids[group_id];
int cur_n_tokens = group_sizes[group_id];          // 本组 Token 数

// 一个 Group 内所有 Token 共享同一份 Expert 权重！
input_ptr  = gathered_input_ptr + group_offsets[group_id] * k;
weight_ptr += expert_id * EXPERT_STRIDE;

// 标准 GEMM（n = cur_n_tokens，可达几十甚至上百）
c_tile = ugemm_gm(weight_ptr, input_ptr, M, cur_n_tokens, K, ...);
```

**Step 2 — Scattered Store（反转 token_map 写回原始顺序）**：
```c
// 不能顺序写！要按照每行对应的原始 Token 位置散射写回
for (int j = 0; j < cur_n_tokens; j++) {
    int orig_token = token_map[offset + sg_j0 + j];   // 查映射表
    row_ptr = out_ptr + slot * n_tokens * M + orig_token * M;
    row_ptr[sg_i0 + i] = c_tile_half.x[...];           // 散射写回
}
```

#### 输出
| 形状 | 说明 |
|:---|:---|
| `[top_k, n_tokens, N]` | 结果恢复到原始 Token 顺序 |

---

## 四、 数据流完整示例

以 `n_tokens=32, top_k=2, n_experts=8, K=4096, N=2048` 为例：

```
原始输入
  A: [1, 32, 4096]   indices: [32, 2]   B: [8, 2048, 4096]

── Kernel B1: bgm_sort ────────────────────────────────────────────────
  indices: [32,2] → 统计 16个(slot,expert) bin 的 Token 数
  输出: GROUP_EXPERT_IDS, GROUP_SIZES, GROUP_OFFSETS, TOKEN_MAP, NUM_GROUPS
  例: Group0=(slot=0, expert=3, size=5, offset=0)
      Group1=(slot=0, expert=7, size=4, offset=5) ...
      NUM_GROUPS = 12

── Kernel B2: bgm_gather ──────────────────────────────────────────────
  TOKEN_MAP[0..4] = [2, 8, 11, 19, 27]  (Group0的5个Token的原始idx)
  将 A[0,2,:], A[0,8,:], A[0,11,:], A[0,19,:], A[0,27,:] 搬运到
       GATHERED_A[0,:], [1,:], [2,:], [3,:], [4,:]
  以此类推所有 Group → GATHERED_A: [64, 4096]

── Kernel B3: gather_matmul_batched ───────────────────────────────────
  Group0: weight_ptr = B[3,:,:]   (Expert 3 权重，加载一次！)
          GEMM: [2048,4096] × [4096,5] → C[2048,5]  (5个Token共享Expert3)
          Scattered Store: C[:,0] → out[0,2,:], C[:,1] → out[0,8,:] ...

  Group1: weight_ptr = B[7,:,:]   (Expert 7 权重，加载一次！)
          GEMM: [2048,4096] × [4096,4] → C[2048,4]
          Scattered Store: C[:,i] → out[0, orig_token, :] ...

最终输出: out[2, 32, 2048]  (top_k=2 个 slot，32 token，2048 特征)
```

---

## 五、 两方案核心代码差异对比

### 5.1 Sort/Mask Kernel：旧方案并行度远高

**旧方案 `moe_mask_gen.cl`**（GPU 波前级并行前缀扫描）：
```c
const size_t expert_id = get_local_id(0);  // n_experts 个线程同时跑！
// ... 每线程统计自己负责的 expert 的 token 数 ...
int tokens_per_expert_iter = work_group_scan_exclusive_add(num_tokens_per_curr_expert);
int experts_id_iter        = work_group_scan_exclusive_add(is_used);
// ↑ GPU 硬件一条指令完成全组前缀和，几乎是 O(1) 时间
```

**新方案 `gathermatmul_sort.cl`**（纯串行）：
```c
// Dispatch: single workgroup (1,1,1) global, (1,1,1) local
// All work done sequentially — n_tokens * top_k is small enough.
wgs.global = {1, 1, 1};   // 完全串行！
for (int s = 0; s < top_k; s++)
    for (int e = 0; e < n_all_experts; e++)
        // O(n_tokens × top_k × n_experts) 的串行循环
```

### 5.2 SwiGLU 处理：旧方案在寄存器内完成，新方案过显存

**旧方案 `moe_gemm.cl`**（融合在 GEMM Kernel 内，零中间显存）：
```c
#ifdef POST_PROC_SILU_MUL
    // c_tile_half 是寄存器变量，SwiGLU 完全在寄存器中完成
    float gate_val = post_op_row[i];                              // 读 Gate 分支输出
    float up_val   = c_tile_half.x[reg_idx_i][reg_idx_j];        // Up 分支在寄存器里
    float res      = gate_val * (up_val / (1.0f + native_exp(-up_val)));  // SiLU(Up) × Gate
    c_tile_half.x[reg_idx_i][reg_idx_j] = res;                    // 不落显存！
#endif
```

**新方案**（SwiGLU 是独立 OP）：
```c
// No POST_PROC_SILU_MUL — activation functions are separate ops in the GatherMatmul graph.
tile_store(c_tile_half, out_ptr, m, cur_n_tokens, sg_i0, sg_j0);
// ↑ Gate 结果写回显存 → SwiGLU OP 从显存读 → 计算 → 写回显存 → 下一个 GatherMatmul 从显存读
```

### 5.3 Scatter/Reduce：旧方案一步完成加权累加

**旧方案 `moe_scatter_reduction_ref.cl`**（乘路由权重 + 多专家累加在一个 Kernel）：
```c
for (int e_iter = 0; e_iter < ACTIVE_EXPERTS; ++e_iter) {
    INPUT2_TYPE weight = expert_weights[token_id * ACTIVE_EXPERTS + e_iter];
    // ...
    if (e_iter == 0)
        output[out_pos + h] = input[in_pos + h] * weight;   // 乘路由分数
    else
        output[out_pos + h] += input[in_pos + h] * weight;  // 多专家累加
}
```

**新方案 `gather_matmul_batched.cl`**（只做位置散射，无加权）：
```c
int orig_token = token_map[offset + sg_j0 + j];
row_ptr = out_ptr + slot * n_tokens * m + orig_token * m;
row_ptr[sg_i0 + i] = c_tile_half.x[...];  // 无权重乘法，无累加
```

---

## 六、 综合优缺点对比

| 维度 | 旧方案（`moe_gemm` 体系） | 新 GatherMatmul 方案 | 胜者 |
|:---|:---|:---|:---:|
| **Sort 并行度** | GPU 并行前缀扫描，近 O(1) | 单线程串行循环，O(n × top_k × n_exp) | **旧方案** |
| **Gather 向量化** | 泛化 `VLOAD/VSTORE`，宽度动态 | 硬编码 `vload8/vstore8` (128-bit)，K+Token 双维并行 | **新方案** |
| **SwiGLU 融合** | `POST_PROC_SILU_MUL` 在寄存器内完成，零中间显存 | 强制拆出为独立 OP，2 次完整显存 Round-trip | **旧方案** |
| **GEMM 计算本体** | 相同的 `ugemm_micro` 底层微内核 | 相同的 `ugemm_micro` 底层微内核 | **持平** |
| **Scatter + 加权 Reduce** | 一个 Kernel 完成乘路由权重 + 多专家累加 | 只做散射，加权需额外独立 OP | **旧方案** |
| **代码可维护性** | 高度耦合，添加新数据类型需大量改动 | 标准 GEMM 接口，直接继承所有 GEMM 优化（int4/fp8） | **新方案** |
| **单/多 Token 统一** | 两套完全独立代码，图层面有分支 | 同一算子，按 n_tokens 自动切换，图层面统一 | **新方案** |
| **算子通用性** | 只能处理固定的 3-GEMM SwiGLU MoE 结构 | 可组合任意激活函数，可服务非 MoE 的分组投影场景 | **新方案** |

---

## 七、 两阶段性能表现分析

### 7.1 Prefill（First Token Latency）

| 方案 | 表现 | 原因 |
|:---|:---:|:---|
| 旧方案（多 Token micro_gemm 路径） | ✅ 好 | GPU 并行前缀扫描完成分组，Grouped GEMM 共享权重 |
| 新方案（GatherMatmul batched） | ✅ 好（相当） | 继承相同的 Grouped GEMM 思想，但 Sort 是串行瓶颈 |

**结论：两者在 Prefill 上基本持平，但新方案的 Sort 串行问题在极大 n_tokens 下会成为可见瓶颈。**

### 7.2 Decode（Token Rate / Second Token Latency）

| 方案 | 表现 | 原因 |
|:---|:---:|:---|
| 旧方案（`moe_3gemm_swiglu_fuse` Megakernel） | ✅✅ 极佳 | 单 Token 走巨型融合 Kernel，Gate/Up/SwiGLU/Down 全在寄存器内一次性完成，零中间显存 I/O，单次 Kernel 启动 |
| 新方案（GatherMatmul per-token） | ⚠️ 一般 | SwiGLU 强制分离，引入中间显存读写；多 Kernel 启动开销 |

**结论：新方案在 Decode 单 Token 场景下有明确的性能代价。**

### 7.3 选型结论

新方案（GatherMatmul）更适合**现代服务化推理场景**（Continuous Batching、大并发、大 Prefill），以及**长远的可维护性和新硬件扩展性**。旧方案的极致单 Token 性能优势仅在严格 BS=1 场景下成立，在实际服务部署中几乎不占主导。

---

## 八、 具体优化建议

### 优化一（High Priority）：`bgm_sort` 改为 GPU 并行前缀扫描

**当前代码**（`gathermatmul_sort_gen.cpp`）：
```cpp
// Single workgroup — all work done sequentially
wgs.global = {1, 1, 1};
wgs.local  = {1, 1, 1};
```

**改造思路**：仿照旧的 `moe_mask_gen`，将 Dispatch 改为 `{n_all_experts × top_k, 1, 1}`，让每个线程负责一个 `(slot, expert)` bin，用 `work_group_scan_exclusive_add` 完成 GPU 硬件级前缀求和。

**预期收益**：Sort 耗时降低 10x+（从 O(n_tokens × top_k × n_experts) 串行降为 O(n_tokens) 并行）。

---

### 优化二（High Priority）：为 `gather_matmul_batched` 添加 SwiGLU 后融合

**问题根源**（`gather_matmul.cl` 注释）：
```c
// No POST_PROC_SILU_MUL — activation functions are separate ops in the GatherMatmul graph.
```

**改造思路**：在 `gather_matmul_batched.cl` 的 Scattered Store 之前，仿照 `moe_gemm.cl` 中的 `POST_PROC_SILU_MUL` 段加入可选的寄存器内激活融合；同时在 `moe_op_fusion` 图变换中识别 `GatherMatmul + SwiGLU` 组合模式，将激活融合标记传入 Kernel。

**预期收益**：消除中间 `[top_k × n_tokens × intermediate_size]` 数据量的 2 次显存 Round-trip，对大 hidden size 模型（如 Deepseek-R1：hidden=5120）效益显著。

---

### 优化三（Medium Priority）：Fused Gather-GEMM，消除 `GATHERED_A` 缓冲区

**问题**：`GATHERED_A` 缓冲区大小 `n_tokens × top_k × K`，在大 Prefill 时可达数十 MB，每次推理全量写入后再全量读出。

**改造思路**：将 `bgm_gather` 和 `gather_matmul_batched` 合并为一个 Kernel。在 GEMM 的 Inner Loop 中直接通过 `token_map` 查找原始 Token 地址取值，绕过 `GATHERED_A` 中间缓冲区。

**权衡**：融合后 Activation 访问变为随机访问，Cache hit rate 会下降。K 值很大时原来连续的 `GATHERED_A` 对 Cache 更友好，实际收益需 Profile 验证。

---

### 优化四（Medium Priority）：`COPY_BLOCK = 8` 动态化

**当前**（`gathermatmul_gather.cl`）：
```c
#define COPY_BLOCK 8  // hardcoded 128-bit vector
half8 val = vload8(0, src + k_offset);
```

**改造思路**：在 `gathermatmul_gather_gen.cpp` 中根据 K 大小和硬件能力动态选择拷贝向量宽度（类似旧方案 `moe_gather_ref.cpp` 的 `GetBlockSize` 逻辑），并探索 `intel_sub_group_block_read/write` 以提升 Cache 利用率。

---

### 优化五（Medium Priority）：`BATCHED_PREFILL_THRESHOLD = 16` 自适应化

**当前**（`gather_matmul.cpp`）：
```cpp
static constexpr int32_t BATCHED_PREFILL_THRESHOLD = 16;  // hardcoded
```

**改造思路**：改为动态计算。最优阈值应满足每 Expert 平均 Token 数达到 GEMM 相比 GEMV 的 Break-even 点（通常 ≥ 4-8）：
```cpp
const int avg_tokens_per_expert = (rtp->n_tokens * rtp->top_k) / n_all_experts;
return avg_tokens_per_expert >= 4;  // 替代固定的 16
```

---

### 优化建议汇总

| 优化项 | 适用阶段 | 预期收益 | 实现难度 |
|:---|:---:|:---:|:---:|
| **Sort 并行化**（P1） | Prefill（大 n_tokens） | Sort 耗时降低 10x+ | 中 |
| **SwiGLU 后融合**（P2） | Decode + Prefill | 减少 2 次完整显存 R/W，省去一个 Kernel 启动 | 高（需图层面配合） |
| **Fused Gather-GEMM**（P3） | Prefill | 消除 `GATHERED_A` 显存读写 | 高（需重构 GEMM 内核） |
| **动态向量宽度**（P4） | Prefill Gather 阶段 | Gather 带宽提升约 1.5-2x | 低 |
| **自适应阈值**（P5） | 全场景 | 避免小 n_tokens 的无效 Sort/Gather 开销 | 低 |

**整体建议**：优先实施 P1（Sort 并行化）消除串行瓶颈 + P2（SwiGLU 融合）消除中间显存浪费，这两项组合对大 Prefill 的吞吐率提升最显著。

---

## 九、 关键文件索引

| 文件 | 功能 |
|:---|:---|
| `src/plugins/intel_gpu/src/graph/impls/ocl_v2/gather_matmul/gather_matmul.cpp` | C++ 侧路径决策与 Kernel 编排 |
| `src/plugins/intel_gpu/src/graph/impls/ocl_v2/gathermatmul_sort.cl` | Sort Kernel（单线程排序，确定分组结构） |
| `src/plugins/intel_gpu/src/graph/impls/ocl_v2/gathermatmul_gather.cl` | Gather Kernel（搬运 Activation 到连续缓冲区） |
| `src/plugins/intel_gpu/src/graph/impls/ocl_v2/gather_matmul.cl` | Per-token GEMV Kernel（Decode/小 Batch） |
| `src/plugins/intel_gpu/src/graph/impls/ocl_v2/gather_matmul_batched.cl` | Batched GEMM + Scattered Store Kernel（大 Batch Prefill） |
| `src/plugins/intel_gpu/src/graph/impls/ocl_v2/gather_matmul/gather_matmul_gen_micro.cpp` | Micro GEMM 生成器（基于 gemmstone） |
| `src/plugins/intel_gpu/src/graph/impls/ocl_v2/moe_mask_gen.cl` | 旧方案 Sort（GPU 并行前缀扫描，对比参考） |
| `src/plugins/intel_gpu/src/graph/impls/ocl_v2/moe_gemm.cl` | 旧方案 GEMM（含 `POST_PROC_SILU_MUL` 融合激活，对比参考） |
| `src/plugins/intel_gpu/src/graph/impls/ocl_v2/moe_3gemm_swiglu_fuse.cl` | 旧方案单 Token 巨型融合 Kernel（对比参考） |
| `src/common/transformations/include/transformations/common_optimizations/moe_op_fusion.hpp` | 图变换：GatherMatmul 序列 → MoE OP 融合（SwiGLU 后融合的扩展点） |
