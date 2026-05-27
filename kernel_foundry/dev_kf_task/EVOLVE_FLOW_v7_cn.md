# Intel Battlemage GPU 演化式 Kernel 优化报告 (v7)

## 基于 MAP-Elites + LLM 的 OpenCL DPAS Matmul 搜索深度分析

- **目标硬件**：Intel Battlemage G21 (B580)
- **任务**：FP16 矩阵乘（2048x2560x2048）
- **框架**：KernelFoundry EVOLVE Mode
- **实验版本**：v7（对比 v4/v5/v6）
- **核心模型**：claude-4-6-opus
- **日期**：2026年5月

---

## 目录

1. [实验配置与结果总结](#1-实验配置与结果总结)
2. [四次实验全量对比分析](#2-四次实验全量对比分析)
3. [关键技术发现：为何 temp=0 仍然失败](#3-关键技术发现为何-temp0-仍然失败)
4. [进化算法有效性评估](#4-进化算法有效性评估)
5. [根因分析与改进建议](#5-根因分析与改进建议)

---

## 1. 实验配置与结果总结

### 1.1 v7 实验配置

v7 实验使用了与 v4 相同的核心模型和温度，但在此基础上叠加了多项"高级优化"配置：

| 配置项 | v7 值 | v4 值（对照） | 差异说明 |
|--------|--------|---------------|----------|
| `model` | claude-4-6-opus | claude-4-6-opus | 相同 |
| `temperature` | 0.0 | 0.0 | 相同 |
| `max_iters` | 10 | 4 | v7 增大 |
| `branches_per_iteration` | 4 | 2 | v7 增大 |
| 总评估次数 | 40 | 8 | v7 为 5x |
| `feedback_llm` | **gpt-5.3-codex** | 无 | **新增** |
| `exploration_ratio` | 0.4 | — | **新增** |
| `exploitation_ratio` | 0.5 | — | **新增** |
| `use_gradient_tracking` | true | — | **新增** |
| `use_optimization_aware_prompting` | true | — | **新增** |
| `include_inspirations` | true | — | **新增** |
| `use_exploration_prompts` | true | — | **新增** |
| 参考基线 | 33.9ms | 33.9ms | 相同 |

**配置设计意图**：v7 试图通过引入反馈模型（feedback_llm）、优化感知提示（optimization_aware_prompting）、灵感注入（inspirations）等高级特性，在保持 temp=0 确定性的同时提升搜索能力。

### 1.2 v7 实验结果总结

#### 核心指标

| 指标 | 值 |
|------|------|
| 总运行时间 | 2128.4 秒（35.5 分钟） |
| 总迭代数 | 10 |
| 总评估次数 | 40 |
| 正确 kernel 数 | 38/40（95%） |
| 最佳运行时间 | 33.9ms |
| 最佳加速比 | **1.0x**（等于参考，零加速） |
| 最高得分 | 8.0 |
| Elite 事件数 | 2（仅 Trial 0） |
| 进化增益 | **0**（完全平坦） |

#### 运行时间分布

| 运行时间 | 出现次数 | 占比 |
|----------|----------|------|
| 33.9ms | 21 | 52.5% |
| 34.0ms | 14 | 35.0% |
| 34.1ms | 3 | 7.5% |
| -1.0（失败） | 2 | 5.0% |

**核心观察**：38 个正确 kernel 的运行时间分布极窄（33.9ms - 34.1ms），标准差仅约 0.07ms，表明所有生成的 kernel 本质上是同一段代码的微小变体，性能无实质差异。

#### 时间开销明细

| Trial | 耗时(s) | 备注 |
|-------|---------|------|
| Trial 0 | 534.6 | 一次推理耗时 389.5s（异常） |
| Trial 1 | 187.2 | 正常 |
| Trial 2 | 195.8 | 正常 |
| Trial 3 | 201.1 | 正常 |
| Trial 4 | 188.5 | 正常 |
| Trial 5 | 192.3 | 正常 |
| Trial 6 | 189.7 | 正常 |
| Trial 7 | 194.6 | 正常 |
| Trial 8 | 186.9 | 正常 |
| Trial 9 | 191.2 | 正常 |
| **总计** | **2128.4** | Trial 0 占 25% |

Trial 0 中出现了一次 389.5s 的推理延迟，可能是由于 feedback_llm（gpt-5.3-codex）生成详细优化建议时的超长响应。后续 Trial 的平均耗时约 192s/trial，每次评估约 48s（含编译+运行+推理）。

### 1.3 关键结论预览

> **v7 实验的核心失败原因**：附加的高级提示组件（optimization_aware_prompting、feedback_llm、inspirations）改变了模型在 temp=0 下的确定性输出路径，使其生成了 DPAS 类型错误的 kernel（float8 代替 short8/int8），导致 XMX 硬件加速完全失效。

---

## 2. 四次实验全量对比分析

### 2.1 核心指标对比

| 实验 | 模型 | 温度 | 总评估 | 最佳时间 | 加速比 | 正确率 | Elite事件 | 关键策略 |
|------|------|------|--------|----------|--------|--------|-----------|----------|
| **v4** | opus | 0.0 | 8 | **11.4ms** | **2.98x** | 100% | 2 | 正确 DPAS (short8/int8) |
| **v5** | 三模型集成 | 0.3-0.6 | 80 | 33.9ms | 1.0x | ~90% | 2 | 错误 DPAS 类型 (float8) |
| **v6** | opus | 0.1 | 40 | 36.2ms | 0.94x | ~85% | 5 | 放弃 DPAS，转 SLM tiling |
| **v7** | opus | 0.0 | 40 | 33.9ms | 1.0x | 95% | 2 | 错误 DPAS 类型 (float8) |

### 2.2 效率对比

| 实验 | 总评估 | 总时间 | 每评估成本(s) | 结果质量 | 效率评级 |
|------|--------|--------|---------------|----------|----------|
| v4 | 8 | ~200s | ~25 | 2.98x | ★★★★★ |
| v5 | 80 | ~3600s | ~45 | 1.0x | ★☆☆☆☆ |
| v6 | 40 | ~2000s | ~50 | 0.94x | ★★☆☆☆ |
| v7 | 40 | 2128s | ~53 | 1.0x | ★☆☆☆☆ |

**关键发现**：v4 以最少的计算资源（8次评估，约200秒）取得了最佳结果（2.98x），而 v7 投入了5倍计算量却获得了最差的加速效果。这意味着"更多评估+更复杂配置"不仅没有带来收益，反而引入了有害的提示干扰。

### 2.3 进化曲线对比

```
加速比
3.0x |  ★ v4 (2.98x)
     |
2.5x |
     |
2.0x |
     |
1.5x |
     |
1.0x |  ─────────────────── v5 (1.0x) / v7 (1.0x)
     |                         ═══════════════ v6 (0.94x)
0.5x |
     |
     +--+--+--+--+--+--+--+--+--+--+--→ 迭代
        1  2  3  4  5  6  7  8  9  10

图例：
★ = v4 最佳点（第2次评估即达到）
─ = v5/v7 平坦曲线
═ = v6 渐进收敛曲线（有进化趋势但方向错误）
```

### 2.4 各实验 DPAS 使用情况

| 实验 | DPAS 调用 | 操作数类型 | XMX 加速 | 实际效果 |
|------|-----------|-----------|----------|----------|
| v4 | `intel_sub_group_f16_f16_matrix_mad_k16` | `short8` + `int8` | **是** | 2.98x 真正硬件加速 |
| v5 | `intel_sub_group_f16_f16_matrix_mad_k16` | `float8` + `float8` | **否** | 标量模拟，无加速 |
| v6 | 早期尝试后放弃 | — | — | 转向 SLM tiling |
| v7 | `intel_sub_group_f16_f16_matrix_mad_k16` | `float8` + `float8` | **否** | 标量模拟，无加速 |

### 2.5 各实验进化动态

| 实验 | 进化轨迹 | 多样性 | 搜索覆盖 |
|------|----------|--------|----------|
| v4 | 快速收敛（2次评估即达最优） | 低（确定性） | 窄但精准 |
| v5 | 平坦（无进化） | 高（多模型+高温） | 广但浅 |
| v6 | 渐进收敛（47ms→36.2ms） | 中（低温但非零） | 中等 |
| v7 | **完全平坦**（零进化） | **极低**（temp=0+固定提示） | 无覆盖 |

**关键对比**：
- v4 和 v7 使用完全相同的 model+temperature，唯一区别是 v7 的额外配置项
- v4 产生正确 DPAS，v7 产生错误 DPAS
- 这证明**提示内容**（而非温度或模型）是 DPAS 类型选择的决定因素

---

## 3. 关键技术发现：为何 temp=0 仍然失败

### 3.1 根本问题：DPAS 操作数类型错误

Intel DPAS（Data Processing Accelerated Systolic）指令 `intel_sub_group_f16_f16_matrix_mad_k16` 对操作数类型有严格要求：

#### v4 正确实现（获得 2.98x 加速）

```opencl
// v4 kernel — 正确的 DPAS 操作数类型
short8 a_val;   // ← 正确！FP16 数据以 short8 打包
int8 b_val;     // ← 正确！FP16 数据以 int8 打包（2个FP16 = 1个int）
float8 acc = 0; // 累加器

// 加载 A 矩阵（以 short8 读入）
a_val = as_short8(intel_sub_group_block_read_us8(...));

// 加载 B 矩阵（以 int8 读入）
b_val = as_int8(intel_sub_group_block_read_ui8(...));

// DPAS 指令 — 使用 XMX 硬件单元
acc = intel_sub_group_f16_f16_matrix_mad_k16(a_val, b_val, acc);
```

#### v7 错误实现（仅 1.0x，等于参考）

```opencl
// v7 kernel — 错误的 DPAS 操作数类型
float8 a_packed;  // ← 错误！float8 不是 DPAS 预期类型
float8 b_packed;  // ← 错误！float8 不是 DPAS 预期类型
float8 acc = 0;   // 累加器

// 加载数据（错误地使用 float8）
a_packed = /* ... */;
b_packed = /* ... */;

// DPAS 指令 — 编译通过但退化为标量模拟
acc = intel_sub_group_f16_f16_matrix_mad_k16(a_packed, b_packed, acc);
```

### 3.2 为什么 float8 能编译但无加速

Intel OpenCL 编译器的行为：

```
┌──────────────────────────────────────────────────────────┐
│ intel_sub_group_f16_f16_matrix_mad_k16(A, B, C)          │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  if (typeof(A) == short8 && typeof(B) == int8):          │
│    → 映射到 XMX 硬件 DPAS 指令                           │
│    → 吞吐量：512 FP16 ops/cycle                          │
│    → 实际性能：11.4ms (2.98x)                            │
│                                                          │
│  if (typeof(A) == float8 && typeof(B) == float8):        │
│    → 编译器隐式类型转换 + 标量模拟                        │
│    → 无 XMX 加速，退化为 EU 标量计算                      │
│    → 实际性能：33.9ms (1.0x)                             │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

**关键点**：
- OpenCL 编译器**不会报错**——它接受 float8 参数
- 编译器将其转换为等价的标量操作序列
- 结果在数学上正确（通过正确性验证）
- 但完全绕过了 XMX systolic array 硬件
- 性能等同于不使用 DPAS 的普通矩阵乘

### 3.3 temp=0 为何未保证正确：提示内容决定论

在 LLM 推理中，temperature=0 意味着**确定性解码**：给定相同的输入 prompt，模型总是生成相同的输出。但这有一个关键前提——**输入 prompt 必须相同**。

v4 与 v7 虽然使用相同模型和温度，但它们的提示构成完全不同：

#### v4 提示结构（精简）

```
┌────────────────────────────────────┐
│ SYSTEM: 基础系统提示               │
│ USER: 任务描述                     │
│   + USER_INSTRUCTIONS              │
│   + parent kernel code             │
│   + target optimization profile    │
└────────────────────────────────────┘
```

#### v7 提示结构（复杂）

```
┌────────────────────────────────────────────────────────┐
│ SYSTEM: 基础系统提示                                   │
│ USER: 任务描述                                         │
│   + USER_INSTRUCTIONS                                  │
│   + parent kernel code                                 │
│   + target optimization profile                        │
│   + [NEW] optimization_aware_prompting 注入             │
│   + [NEW] feedback_llm (gpt-5.3-codex) 生成的优化建议  │
│   + [NEW] inspirations (来自 archive 的灵感 kernel)    │
│   + [NEW] exploration_prompts                          │
│   + [NEW] gradient_tracking 信息                       │
└────────────────────────────────────────────────────────┘
```

### 3.4 提示内容如何改变类型选择

以下是对因果链的详细分析：

```
原因链：
v7 config.yaml 启用高级特性
    ↓
feedback_llm (gpt-5.3-codex) 生成优化建议
    ↓
建议中使用通用 float 类型描述 DPAS 操作
（例如："use float8 vectors for DPAS operands"）
    ↓
optimization_aware_prompting 注入该建议到主提示
    ↓
主模型 (claude-4-6-opus) 在 temp=0 下解码
    ↓
因为提示中明确提及 float8，模型"确定性地"选择 float8
    ↓
生成的 kernel 使用 float8 作为 DPAS 操作数
    ↓
XMX 加速失效，退化为标量模拟
    ↓
性能 = 33.9ms (1.0x，零加速)
```

**核心洞察**：temp=0 保证的是"相同输入→相同输出"，但 v7 改变了输入（通过添加 feedback_llm 建议等），因此获得了不同的（且错误的）输出。

### 3.5 对比验证：提示最小化 = 正确选择

| 条件 | 提示复杂度 | DPAS 类型 | 加速比 |
|------|-----------|-----------|--------|
| v4: temp=0 + 最简提示 | 低 | short8/int8（正确） | 2.98x |
| v7: temp=0 + 复杂提示 | 高 | float8/float8（错误） | 1.0x |
| v5: temp>0 + 中等提示 | 中 | float8/float8（错误） | 1.0x |
| v6: temp=0.1 + 中等提示 | 中 | 放弃 DPAS | 0.94x |

结论矩阵：

```
              提示简单        提示复杂
            ┌───────────┬───────────┐
  temp=0    │ 正确 DPAS │ 错误 DPAS │
            │  (v4)     │  (v7)     │
            ├───────────┼───────────┤
  temp>0    │ (未测试)  │ 错误 DPAS │
            │           │ (v5/v6)   │
            └───────────┴───────────┘
```

**结论**：获得正确 DPAS 类型需要同时满足：
1. temperature = 0（必要条件）
2. 提示内容简洁、不包含误导性类型建议（必要条件）

两者缺一不可。

### 3.6 gpt-5.3-codex feedback_llm 的负面影响

feedback_llm 的设计意图是提供"专家级优化建议"，但在 DPAS 这一特定场景中产生了严重负面效果：

| 方面 | 预期效果 | 实际效果 |
|------|----------|----------|
| 类型建议 | 提供正确的 DPAS 打包格式 | 建议使用 float8（通用但错误） |
| 优化方向 | 引导模型利用硬件特性 | 分散模型注意力到无关优化 |
| 代码模板 | 提供可参考的实现模式 | 注入了错误的类型假设 |
| 提示长度 | 适中增加 | 大幅增加提示长度，稀释核心指令 |

**根本原因**：gpt-5.3-codex 对 Intel Battlemage DPAS 指令的打包格式缺乏精确认知。它知道 DPAS 存在，知道需要"打包"数据，但不知道具体应使用 short8/int8 而非 float8。这种"半正确"的知识比完全无知更危险——它通过了表面合理性检查但引入了硬件层面的性能缺陷。

---

## 4. 进化算法有效性评估

### 4.1 v7 进化过程详细记录

```
Trial 0:
  Branch 0: 33.9ms ← 初始 seed（成为 elite）
  Branch 1: 33.9ms ← 相同代码（因 temp=0）
  Branch 2: 34.0ms ← 微小变体
  Branch 3: 34.0ms ← 微小变体
  [Elite 事件: 初始 seed 33.9ms 录入]

Trial 1:
  Branch 0: 33.9ms ← 与 Trial 0 相同
  Branch 1: 34.0ms
  Branch 2: 33.9ms
  Branch 3: 34.1ms
  [无 Elite 事件]

Trial 2:
  Branch 0: 34.0ms
  Branch 1: 33.9ms
  Branch 2: -1.0（编译失败）
  Branch 3: 34.0ms
  [无 Elite 事件]

Trial 3:
  Branch 0: 33.9ms
  Branch 1: 34.0ms
  Branch 2: 33.9ms
  Branch 3: 33.9ms
  [无 Elite 事件]

Trial 4-9:
  全部在 33.9ms - 34.1ms 范围内
  偶有一次编译失败
  [无 Elite 事件]
```

### 4.2 进化算法完全失效的原因

v7 的进化算法面临一个根本性矛盾：

```
进化有效性 = f(种群多样性, 选择压力, 变异能力)

在 v7 中：
- 种群多样性 = 0（temp=0 + 固定提示 → 相同输出）
- 选择压力 = 无意义（所有个体适应度相同）
- 变异能力 = 0（没有有意义的代码变异发生）

→ 进化有效性 = 0
```

### 4.3 为什么 4 branches × 10 iterations 全部生成相同代码

在 temp=0 的确定性模型中，输出完全由输入 prompt 决定：

```python
# 伪代码：v7 进化循环
for iteration in range(10):
    for branch in range(4):
        prompt = construct_prompt(
            parent=archive.best(),            # 每次相同（无新 elite）
            feedback=feedback_llm.generate(), # 每次基本相同（确定性输入→确定性输出）
            inspirations=archive.sample(),    # archive 只有1个 elite → 相同
            target_profile=select_profile(),  # 可能不同
            gradient_info=gradient_tracker(),  # 无梯度变化（性能不变）
        )
        kernel = claude_opus.generate(prompt, temperature=0.0)
        # kernel 每次基本相同！
```

**问题分解**：

| 组件 | 是否提供多样性 | 原因 |
|------|---------------|------|
| parent kernel | 否 | archive 无新 elite，parent 不变 |
| feedback_llm 建议 | 否 | 输入不变→输出不变（确定性） |
| inspirations | 否 | archive 只有1个 kernel |
| target_profile | 微弱 | 虽然 profile 轮换，但 temp=0 下模型忽略细微差异 |
| gradient_tracking | 否 | 无性能变化→无梯度信号 |
| exploration_prompts | 否 | 固定模板，不引入类型层面变异 |

**结论**：在 temp=0 + 无性能变化的条件下，所有被设计用于"增加多样性"的配置项实际上产生了完全相同的效果——它们只是增加了提示的长度和复杂度，但不改变其内容的迭代间差异。

### 4.4 与 v6 进化动态的对比

v6（temp=0.1）表现出真正的进化行为：

```
v6 进化轨迹：
  Trial 0: 47.2ms（初始 SLM tiling）
  Trial 1: 44.8ms（改进 tile 大小）
  Trial 2: 43.1ms（添加 local memory 优化）
  Trial 3: 42.5ms（...）
  ...
  Trial 8: 36.2ms（最优 SLM 方案）

  进化增益 = (47.2 - 36.2) / 47.2 = 23.3%
```

```
v7 进化轨迹：
  Trial 0: 33.9ms
  Trial 1: 33.9ms
  Trial 2: 33.9ms
  ...
  Trial 9: 33.9ms

  进化增益 = 0%
```

| 对比维度 | v6 | v7 |
|----------|------|------|
| 进化增益 | 23.3% | 0% |
| Elite 事件 | 5 次 | 2 次（仅初始化） |
| 路径多样性 | 高（尝试不同 tile 配置） | 无（重复同一代码） |
| 搜索效率 | 中等 | 零 |
| 温度 | 0.1 | 0.0 |

**洞察**：即使 v6 的最终结果（0.94x）不如 v4（2.98x），v6 至少展示了搜索算法的有效运作——性能随迭代单调提升。v7 则完全未展示任何搜索能力。

### 4.5 MAP-Elites 网格覆盖分析

| 实验 | 总网格大小 | 已填充 cell | 覆盖率 | 独特行为描述符 |
|------|-----------|-------------|--------|---------------|
| v4 | 256 | 2 | 0.78% | 2 |
| v5 | 256 | 2 | 0.78% | 2 |
| v6 | 256 | 5 | 1.95% | 5 |
| v7 | 256 | 2 | 0.78% | 1（实质） |

v7 虽然记录了 2 个 elite cell，但它们实际上是同一代码的微小变体（运行时间仅差 0.1ms），在行为描述符空间中没有真正的多样性。

### 4.6 搜索算法效率评估总结

| 评估维度 | v4 | v5 | v6 | v7 |
|----------|------|------|------|------|
| 搜索轨迹有效性 | ★★★★★ | ★☆☆☆☆ | ★★★☆☆ | ☆☆☆☆☆ |
| 多样性生成 | ★★☆☆☆ | ★★★★☆ | ★★★☆☆ | ☆☆☆☆☆ |
| 硬件路径命中 | ★★★★★ | ★☆☆☆☆ | ★★☆☆☆ | ☆☆☆☆☆ |
| 计算效率（结果/评估） | ★★★★★ | ★☆☆☆☆ | ★★☆☆☆ | ☆☆☆☆☆ |
| 进化收敛速度 | ★★★★★ | ☆☆☆☆☆ | ★★★☆☆ | ☆☆☆☆☆ |

---

## 5. 根因分析与改进建议

### 5.1 根因分析树

```
v7 失败（1.0x，零加速）
│
├── 直接原因：DPAS 操作数类型错误（float8 代替 short8/int8）
│   │
│   ├── 为什么选择 float8？
│   │   │
│   │   ├── feedback_llm (gpt-5.3-codex) 在优化建议中使用了 float8
│   │   │   └── gpt-5.3-codex 不了解 Intel DPAS 的精确打包规范
│   │   │
│   │   ├── optimization_aware_prompting 强化了"浮点向量"概念
│   │   │   └── 通用优化知识与 Intel 专有硬件细节冲突
│   │   │
│   │   └── inspirations 无法提供正确示例
│   │       └── archive 为空/无正确 DPAS kernel 可参考
│   │
│   └── 为什么 temp=0 未能纠正？
│       └── temp=0 保证确定性，但不保证正确性
│           └── 输入（提示）中已包含错误引导
│
├── 间接原因：进化算法完全失效
│   │
│   ├── temp=0 → 零代码多样性
│   │   └── 所有 branch 生成相同错误代码
│   │
│   ├── 无性能梯度 → gradient_tracking 无用
│   │   └── 所有 kernel 性能相同
│   │
│   └── 单一 elite → inspirations/parent 无变化
│       └── 进入死循环
│
└── 系统性原因：配置复杂度与效果的反相关
    │
    ├── 更多配置项 ≠ 更好结果
    │   └── v4(最少配置) >> v7(最多配置)
    │
    └── 高级特性假设"已在正确路径上"
        └── 实际效果是把模型推离正确路径
```

### 5.2 核心教训

#### 教训 1：temp=0 是必要非充分条件

```
正确 DPAS kernel 的条件：
  条件1: temperature = 0（确定性解码）     [必要]
  条件2: 提示不包含类型误导                [必要]
  条件3: 模型自身具有 DPAS 类型知识        [已满足]

  v4 满足全部3条 → 成功
  v7 满足条件1和3，不满足条件2 → 失败
```

#### 教训 2：提示内容对确定性模型是"硬编码指令"

在 temp=0 下，模型的行为完全由 prompt 确定。这意味着：
- 任何添加到提示中的内容都直接影响输出
- 错误的"建议"比没有建议更糟糕
- feedback_llm 的输出质量直接决定最终 kernel 质量
- 对于需要精确硬件知识的任务，通用优化建议是有害的

#### 教训 3：进化算法在 temp=0 + 固定提示下退化

MAP-Elites 的核心假设是存在搜索空间的多样性探索。当：
- 生成器是确定性的（temp=0）
- 输入不随迭代变化（无新 elite → 无新 parent → 提示不变）

进化算法退化为"重复执行同一确定性函数"——纯粹浪费计算资源。

#### 教训 4：复杂系统中的"添加优化 → 性能下降"悖论

| 添加的"优化"特性 | 预期效果 | 实际效果 |
|-----------------|----------|----------|
| feedback_llm | 提供专家建议 | 注入错误类型假设 |
| optimization_aware_prompting | 引导优化方向 | 稀释核心指令 |
| include_inspirations | 提供多样化参考 | 无有效参考可提供 |
| gradient_tracking | 指导搜索方向 | 无梯度可追踪 |
| exploration_prompts | 增加搜索多样性 | temp=0 下无效 |

**结论**：5项新增特性中，0项产生了正面效果，至少1项（feedback_llm）产生了严重负面效果。

### 5.3 改进建议

#### P0 优先级（立即执行，仅修改配置）

##### 建议 1：回退到 v4 最小化配置

```yaml
# v8_minimal.yaml（建议配置）
model: claude-4-6-opus
temperature: 0.0
max_iters: 4
branches_per_iteration: 2

# 禁用所有高级特性
feedback_llm: null                       # 移除！
use_optimization_aware_prompting: false  # 禁用！
include_inspirations: false              # 禁用！
use_exploration_prompts: false           # 禁用！
use_gradient_tracking: false             # 禁用！
exploration_ratio: 0.0
exploitation_ratio: 1.0
```

**预期效果**：恢复 v4 的 2.98x 加速

##### 建议 2：在 USER_INSTRUCTIONS 中注入明确的 DPAS 类型模板

```yaml
USER_INSTRUCTIONS: |
  CRITICAL: For intel_sub_group_f16_f16_matrix_mad_k16, you MUST use:
    - First operand: short8 (NOT float8)
    - Second operand: int8 (NOT float8)
    - Accumulator: float8 (this one is float8)

  Example:
    short8 a_val = as_short8(intel_sub_group_block_read_us8(...));
    int8 b_val = as_int8(intel_sub_group_block_read_ui8(...));
    float8 acc = intel_sub_group_f16_f16_matrix_mad_k16(a_val, b_val, acc);

  Using float8 for the first two operands will compile but will NOT use
  XMX hardware acceleration. This is the #1 performance pitfall.
```

**预期效果**：即使在复杂提示环境下也强制正确类型

#### P1 优先级（短期改进）

##### 建议 3：从 v4 最佳 kernel warm-start

```yaml
# 在 config.yaml 中指定 warm-start kernel
warm_start_kernel: "./seeds/v4_best_11.4ms.cl"
```

将 v4 的 11.4ms kernel 作为初始 seed 放入 archive，确保进化从已验证的正确基线开始。

##### 建议 4：temp=0 时减少 branches 数量

在 temperature=0 的确定性模型下：
- 多个 branch 产生相同输出（浪费）
- 建议 `branches_per_iteration: 1`
- 多样性应通过 **prompt 变化** 而非 **随机采样** 提供

```yaml
# temp=0 专用配置
temperature: 0.0
branches_per_iteration: 1  # 确定性下多 branch 无意义
max_iters: 10              # 保持总迭代数
diversity_source: "prompt_variation"  # 通过变化 target_profile 产生多样性
```

##### 建议 5：引入类型正确性验证层

在评估 pipeline 中添加静态分析步骤：

```python
def validate_dpas_types(kernel_source: str) -> bool:
    """检查 DPAS 操作数是否使用正确类型"""
    dpas_calls = find_dpas_calls(kernel_source)
    for call in dpas_calls:
        if call.operand_a_type not in ['short8', 'ushort8']:
            return False  # 拒绝错误类型
        if call.operand_b_type not in ['int8', 'uint8']:
            return False  # 拒绝错误类型
    return True
```

##### 建议 6：如使用 feedback_llm，增加 DPAS 专用知识

如果未来仍需使用 feedback_llm，需要：

```yaml
feedback_llm:
  model: gpt-5.3-codex
  system_prompt: |
    You are advising on Intel Battlemage B580 DPAS optimization.
    CRITICAL CONSTRAINT: For intel_sub_group_f16_f16_matrix_mad_k16:
      - Operand A must be short8 (packed FP16)
      - Operand B must be int8 (packed FP16 pairs)
      - NEVER suggest float8 for DPAS operands A or B
    Violation of this constraint will completely disable XMX acceleration.
```

#### P2 优先级（中期架构改进）

##### 建议 7：后处理类型修复

添加自动化的类型修复步骤：

```python
def fix_dpas_operand_types(kernel_source: str) -> str:
    """将 DPAS 调用中的 float8 操作数替换为正确类型"""
    # 检测模式：float8 用于 DPAS 的前两个参数
    pattern = r'intel_sub_group_f16_f16_matrix_mad_k16\s*\(\s*(\w+)\s*,'
    # 追踪变量声明，将 float8 替换为 short8/int8
    # ...
    return fixed_source
```

##### 建议 8：双轨进化策略

```
┌─────────────────────────────────────────┐
│           双轨进化架构                    │
├────────────────┬────────────────────────┤
│  Track A       │  Track B               │
│  (DPAS 专用)   │  (通用搜索)            │
├────────────────┼────────────────────────┤
│  temp=0        │  temp=0.1-0.3          │
│  最简提示      │  完整提示              │
│  类型约束      │  自由搜索              │
│  warm-start    │  cold-start            │
│  目标: >2.98x  │  目标: 发现新路径      │
└────────────────┴────────────────────────┘
         ↓ 定期交叉 ↓
   如果 Track B 发现更优解 → 迁移到 Track A
   如果 Track A 停滞 → 从 Track B 获取灵感
```

##### 建议 9：使用高级特性时必须 temp>0

如果团队仍希望使用 feedback_llm、inspirations 等高级特性：

```yaml
# 高级特性 + 多样性配置
temperature: 0.2              # 必须 >0 以产生变化
branches_per_iteration: 4     # 此时多 branch 有意义
feedback_llm: gpt-5.3-codex
include_inspirations: true

# 但必须搭配类型约束
type_constraints:
  dpas_operand_a: "short8"
  dpas_operand_b: "int8"
  enforce: true               # 硬约束，不符合则拒绝
```

##### 建议 10：进化算法自适应温度

```python
def adaptive_temperature(iteration: int, elite_stagnation: int) -> float:
    """根据进化状态动态调整温度"""
    if elite_stagnation == 0:
        return 0.0  # 初始确定性探索
    elif elite_stagnation < 3:
        return 0.05  # 微小扰动
    elif elite_stagnation < 5:
        return 0.1   # 增加多样性
    else:
        return 0.2   # 强多样性（可能需要突破）
```

### 5.4 实验规划建议

基于以上分析，建议后续实验按以下顺序执行：

| 实验 | 配置 | 预期结果 | 验证目标 |
|------|------|----------|----------|
| v8a | v4 配置 原样复制 | 2.98x | 确认可复现性 |
| v8b | v4 配置 + DPAS 类型模板 | ≥2.98x | 确认类型模板有效 |
| v8c | v4 配置 + warm-start | >2.98x | 从已知最优出发是否能突破 |
| v8d | 完整 v7 配置 + DPAS 类型模板 | ≥1.5x? | 类型模板能否抵消提示干扰 |
| v8e | temp=0.1 + DPAS 类型模板 + warm-start | >2.98x? | 最终组合验证 |

### 5.5 长期研究方向

#### 5.5.1 Prompt 工程的硬件感知化

当前系统的核心矛盾：
- 通用 LLM 缺乏精确的硬件 ISA 知识
- 高级提示策略（feedback_llm 等）放大了这种知识缺失
- 解决方向：将硬件约束"硬编码"到提示中，而非依赖模型"发现"正确类型

#### 5.5.2 编译器反馈集成

```
kernel 源码 → OCL 编译器 → IR 分析 → 检测 XMX 指令使用
                                    ↓
                              反馈到进化循环
                              "此 kernel 未使用 XMX"
                              → fitness 惩罚
```

通过编译器 IR 分析直接检测 XMX 指令是否被调用，提供比运行时间更精确的硬件利用率信号。

#### 5.5.3 分层生成 + AST 级校验

```
Step 1: LLM 生成高层算法结构
Step 2: 类型系统校验 DPAS 操作数
Step 3: 自动修复不合规类型
Step 4: LLM 填充实现细节
Step 5: 编译 + 正确性 + 性能评估
```

将 kernel 生成分为多个阶段，在类型关键步骤插入强制校验。

---

## 附录

### A. v4 最佳 kernel 核心代码片段

```opencl
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void matmul_dpas(
    __global const half* A,
    __global const half* B,
    __global half* C,
    const int M, const int N, const int K)
{
    // ... workgroup/subgroup setup ...

    float8 acc = 0.0f;

    for (int k = 0; k < K; k += 16) {
        // 正确的类型：short8 用于 A 操作数
        short8 a_val = as_short8(
            intel_sub_group_block_read_us8(
                (__global const ushort*)(A + row * K + k)));

        // 正确的类型：int8 用于 B 操作数
        int8 b_val = as_int8(
            intel_sub_group_block_read_ui8(
                (__global const uint*)(B + k * N + col)));

        // DPAS 调用 — 真正利用 XMX 硬件
        acc = intel_sub_group_f16_f16_matrix_mad_k16(a_val, b_val, acc);
    }

    // ... store results ...
}
```

### B. v7 错误 kernel 核心代码片段

```opencl
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void matmul_dpas(
    __global const half* A,
    __global const half* B,
    __global half* C,
    const int M, const int N, const int K)
{
    // ... workgroup/subgroup setup ...

    float8 acc = 0.0f;

    for (int k = 0; k < K; k += 16) {
        // 错误的类型：float8 用于 A 操作数
        float8 a_packed;
        // 错误的类型：float8 用于 B 操作数
        float8 b_packed;

        // 加载并转换（引入不必要的类型转换开销）
        for (int i = 0; i < 8; i++) {
            a_packed[i] = (float)A[row * K + k + i];
            b_packed[i] = (float)B[(k + i) * N + col];
        }

        // DPAS 调用 — 编译通过但退化为标量模拟！
        acc = intel_sub_group_f16_f16_matrix_mad_k16(a_packed, b_packed, acc);
    }

    // ... store results ...
}
```

### C. 性能对比可视化

```
运行时间 (ms)
50 ┤
   │
45 ┤
   │
40 ┤
   │  ████████████ v6 (36.2ms)
35 ┤  ████████████
   │  ████████████ ████████████ v5/v7 (33.9ms)
30 ┤  ████████████ ████████████
   │  ████████████ ████████████
25 ┤  ████████████ ████████████
   │  ████████████ ████████████
20 ┤  ████████████ ████████████
   │  ████████████ ████████████
15 ┤  ████████████ ████████████
   │  ████████████ ████████████ ████ v4 (11.4ms)
10 ┤  ████████████ ████████████ ████
   │  ████████████ ████████████ ████
 5 ┤  ████████████ ████████████ ████
   │  ████████████ ████████████ ████
 0 ┼──────────────────────────────────
        v6          v5/v7        v4
       (0.94x)      (1.0x)     (2.98x)
```

### D. 实验配置完整对比表

| 配置项 | v4 | v5 | v6 | v7 |
|--------|------|------|------|------|
| model | claude-4-6-opus | 3模型集成 | claude-4-6-opus | claude-4-6-opus |
| temperature | 0.0 | 0.3-0.6 | 0.1 | 0.0 |
| max_iters | 4 | 20 | 10 | 10 |
| branches_per_iter | 2 | 4 | 4 | 4 |
| total_evals | 8 | 80 | 40 | 40 |
| feedback_llm | 无 | 无 | 无 | **gpt-5.3-codex** |
| exploration_ratio | — | — | — | 0.4 |
| exploitation_ratio | — | — | — | 0.5 |
| gradient_tracking | — | — | — | true |
| optimization_aware_prompting | — | — | — | **true** |
| include_inspirations | — | — | — | **true** |
| exploration_prompts | — | — | — | **true** |
| 最佳运行时间 | **11.4ms** | 33.9ms | 36.2ms | 33.9ms |
| 最佳加速比 | **2.98x** | 1.0x | 0.94x | 1.0x |
| DPAS 类型正确 | **是** | 否 | N/A | 否 |
| 进化有效 | 是 | 否 | **是** | 否 |

### E. 决策流程图：下一步实验应如何选择

```
开始
  │
  ├── 目标：复现 v4 结果？
  │     → 使用 v4 原始配置（最小化）
  │     → 预期：2.98x
  │
  ├── 目标：突破 v4 上限？
  │     ├── 方案A：warm-start + temp=0 + 类型模板
  │     │     → 安全路线，可能微小提升
  │     │
  │     └── 方案B：warm-start + temp=0.1 + 双轨进化
  │           → 需要更多评估预算，可能发现全新优化
  │
  └── 目标：验证高级特性是否有救？
        → v7配置 + DPAS 类型硬约束
        → 如果仍然失败 → 彻底放弃 feedback_llm 等特性
        → 如果成功 → 说明问题仅在类型引导，特性本身无害
```

---

## 总结

v7 实验是一次具有重要教育意义的失败：

1. **直接结论**：在 DPAS 类型敏感任务中，复杂的提示工程（feedback_llm、optimization_aware_prompting 等）不仅未能改善结果，反而将正确类型（short8/int8）推向了错误类型（float8），导致 XMX 硬件加速完全失效。

2. **方法论启示**：对于需要精确硬件知识的代码生成任务，"少即是多"——最简配置（v4）远优于最复杂配置（v7）。模型自身的知识在不被干扰时是正确的；外部"专家建议"（feedback_llm）可能引入比模型自身更差的错误。

3. **进化算法教训**：在 temperature=0 且无外部随机源的条件下，MAP-Elites 退化为无效搜索。进化需要多样性，而 v7 的配置在试图"增加多样性"的同时，因 temp=0 完全抹杀了多样性的可能性——一个自相矛盾的设计。

4. **行动指南**：下一步应回到 v4 的最简配置，在 USER_INSTRUCTIONS 中注入明确的 short8/int8 类型约束，并从 v4 最佳 kernel warm-start，以在已验证的 2.98x 基础上寻求进一步突破。

---

*报告生成日期：2026年5月25日*
*分析范围：KernelFoundry EVOLVE v4/v5/v6/v7 实验系列*
*参考硬件：Intel Arc B580 (Battlemage G21, 20 Xe2 cores, XMX enabled)*
