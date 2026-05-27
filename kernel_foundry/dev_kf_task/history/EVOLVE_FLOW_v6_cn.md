# Intel Battlemage GPU 演化式 Kernel 优化报告

## 基于 MAP-Elites 的 LLM 驱动 OpenCL DPAS Matmul 搜索分析

- **目标硬件**：Intel Battlemage G21 (B580)
- **任务**：FP16 矩阵乘（2048x2560x2048）
- **框架**：KernelFoundry EVOLVE Mode
- **日期**：2026年5月

---

## 1. 引言

本文分析了一个“演化算法 + LLM 代码生成”的自动优化系统，核心由以下模块组成：

- MAP-Elites（质量-多样性搜索）
- 岛模型并行进化
- QD 梯度追踪
- LLM 语义变异（生成候选 kernel）

目标是在 Intel B580 上优化 FP16 matmul，重点利用 **DPAS/XMX** 硬件能力获得更高吞吐。

本次对比了 v4 / v5 / v6 三组实验，用于分析模型选择与温度参数对收敛和性能的影响。

---

## 2. 进化算法架构

### 2.1 主循环

每轮迭代执行：

1. **采样**：从 MAP-Elites 存档中选择 parent/inspirations
2. **定向**：选择目标优化 profile（mutate/diversify）
3. **构造提示**：将 parent、反馈、目标 profile 注入 prompt
4. **生成**：LLM 并行生成多个分支候选
5. **评估**：编译 -> 正确性 -> 性能测试
6. **更新**：按适应度更新 archive / island

### 2.2 4维行为网格（MAP-Elites）

| 维度 | 取值 | 含义 |
|---|---|---|
| `memory_opt` | 0-3 | 内存层次优化程度 |
| `compute_opt` | 0-3 | 计算策略效率 |
| `parallelism_opt` | 0-3 | 并行粒度 |
| `esimd_opt` | 0-3 | ESIMD/DPAS 使用级别 |

总网格大小：$4\times4\times4\times4=256$ 个 cell。

### 2.3 适应度函数

$$
\text{combined\_score} = \text{perf\_score} + 3 \cdot I(\text{correct} \land \text{speedup}>0) \cdot \text{runtime\_improvement}
$$

其中 `runtime_improvement = reference_runtime / kernel_runtime`。

---

## 3. 实验结果

### 3.1 总览

| 实验 | 模型 | 温度 | 迭代x分支 | 总评估 | 最佳耗时 | 最佳加速 |
|---|---|---:|---:|---:|---:|---:|
| v4 | claude-4-6-opus | 0.0 | 4 x 2 | 8 | **11.4 ms** | **2.98x** |
| v5 | 三模型集成 | 0.3-0.6 | 20 x 4 | 80 | 33.9 ms | 1.0x |
| v6 | claude-4-6-opus | 0.1 | 10 x 4 | 40 | 36.2 ms | 0.94x |

### 3.2 v4（单模型，温度0.0）

- 很快命中高质量 DPAS 实现
- 操作数编码正确（`short8` / `int8`）
- 仅 8 次评估达到 **2.98x**

### 3.3 v5（多模型，高温）

- 正确率高，但性能基本等于参考
- 常见问题：DPAS 调用“可编译但不高效”，未真正发挥 XMX
- archive 覆盖率低（2/256）

### 3.4 v6（单模型，温度0.1）

- 出现真实收敛趋势：约 47 ms -> 43 ms -> **36.2 ms**
- 早期 DPAS 失败后转向 SLM tiling 路径
- 搜索有效，但停在局部最优，未超过参考

---

## 4. 关键结论

1. **温度对 DPAS 正确性极其敏感**
   - `temp=0.0` 更稳定地产生正确 DPAS 编码。
   - `temp>0` 的微小随机性足以扰乱 DPAS 路径。

2. **这类高精度硬件任务中，模型多样性不一定带来收益**
   - 集成提高了“形式正确率”，但削弱了“硬件语义正确率”。

3. **路径依赖明显**
   - 初始失败会把进化轨迹锁定到“更安全但上限更低”的方案。

4. **当前 fitness 存在盲区**
   - 难以表达“短期失败但长期潜力高”的 DPAS 候选价值。

---

## 5. 改进建议

### 5.1 立即可做（仅配置）

- DPAS 任务优先使用 `claude-4-6-opus` + **temperature=0.0**
- 从 v4 已验证最优 kernel warm-start
- 保持单强模型，探索多样性通过 prompt/target profile 提供

### 5.2 短期（小改动）

- 在 `USER_INSTRUCTIONS` 注入强制 DPAS 模板
- 增加 tabu/访问惩罚，减少目标 profile 重复
- 改善 subgroup 风格 kernel 的 host launch 兼容性

### 5.3 中期（架构增强）

- 分阶段温度策略（先稳后探索）
- potential-aware fitness（对 DPAS 路径加潜力奖励）
- DPAS 与非 DPAS 双轨并行进化，支持迁移

### 5.4 长期（研究方向）

- surrogate 预筛选
- profiler 引导反馈（如 XMX 利用率）
- 分层生成与 AST 级校验/自动修复

---

## 6. 结论

进化框架本身是有效的，能收敛；但最终性能上限高度依赖模型是否进入并保持在正确的 **DPAS/XMX** 路径。

对这类任务，当前最优实践是：

- 强单模型
- 确定性生成（`temperature=0.0`）
- 从已验证 DPAS 解 warm-start

这一路线最有希望在现有基础上继续突破性能上限。
