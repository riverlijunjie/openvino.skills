# EVOLVE Mode Flow 深度分析 v5 — 短期优化参数实验

## 1) 执行信息与本次运行结论

| 项目 | 值 |
|------|-----|
| 脚本 | `/mnt/river/kernel_foundry/kernelfoundry.internal/runme_matmul.sh` |
| 目标任务 | `/mnt/river/kernel_foundry/workspace/ocl_matmul_dpas` (OpenCL DPAS matmul) |
| 执行环境 | `conda activate kernel_intel` + `source /opt/intel/oneapi/2025.0/oneapi-vars.sh` |
| GPU | Intel Battlemage G21 (B580), device_id=0xe20b |
| LLM 模型 | 多模型集成: `claude-4-6-opus`(30次) + `claude-4-5-sonnet`(29次) + `gpt-5.3-codex`(21次) |
| `max_iters` | **20** (v4为4) |
| `branches_per_iteration` | **4** (v4为2) |
| 总评估 kernel 数 | **80** (v4为8) |
| 总耗时 | **3508.5 秒 (58.5 分钟)** (v4为269.6秒) |
| 参考基线 (reference) | 34.0 ms (实测33.9ms) |
| 最佳候选 | **33.9 ms** (speedup = **1.0x**，未超越参考) |
| v4对比 | v4最佳为 **11.4 ms** (speedup = **2.98x**) |

> **核心结论：本次实验效果严重倒退。80个候选kernel中无一超越参考基线，而v4仅用8个候选即获得2.98x加速。参数调优方向选择正确但实施产生了副作用——多模型集成引入了能力较弱的模型，稀释了最强模型(claude-4-6-opus)独占时的优势。**

---

## 2) 配置变更对比 (v4 → v5)

### 2.1 实际生效配置 (从 `workspace/ocl_matmul_dpas/config.yaml`)

| 参数 | v4 值 | v5 值 | 预期效果 | 实际效果 |
|------|-------|-------|----------|----------|
| `max_iters` | 4 | **20** | 更多搜索 | ✓ 更多样本 |
| `branches_per_iteration` | 2 | **4** | 更宽搜索 | ✓ 更多样本 |
| `inference.servers` | claude-4-6-opus 单模型 | **3模型集成** | 代码多样性 | ✗ 引入弱模型 |
| `temperature` | 0.0 | **0.3-0.6** | 随机性 | ✗ 降低精确度 |
| `exploration_ratio` | 0.2 | **0.4** | 更多探索 | 中性 |
| `exploitation_ratio` | 0.7 | **0.5** | 减少利用 | 中性 |
| `use_feedback_llm` | false | **true** | 优化建议 | 不明确 |
| `gradient_config.exploration_weight` | 0.2 | **0.4** | 更多探索 | 中性 |

### 2.2 模型配置详情

```yaml
# v5 实际使用的推理配置 (task config.yaml)
inference:
  servers:
  - model_name: claude-4-5-sonnet   # temperature: 0.6
  - model_name: gpt-5.3-codex       # temperature: 0.4
  - model_name: claude-4-6-opus     # temperature: 0.3
```

vs v4 使用 `configs/inference/server.yaml` 默认:
```yaml
# v4 实际使用
servers:
  - model_name: claude-4-6-opus     # temperature: 0.0
```

---

## 3) 全量评估结果

### 3.1 总体统计

| 指标 | 值 |
|------|-----|
| 总评估数 | 80 (20 trials × 4 branches) |
| 编译成功 | 80/80 (100%) |
| 正确性通过 | 76/80 (95%) |
| 超越参考 | **0/80 (0%)** |
| 匹配参考 (33.9ms) | 70/80 (87.5%) |
| 略慢 (34.0ms) | 7/80 (8.75%) |
| 编译但不正确 | 2/80 (2.5%) |
| 明显慢 (145ms) | 1/80 (1.25%) |

### 3.2 逐 Trial 最佳结果

| Trial | Best Runtime (ms) | Speedup | Scores [v0,v1,v2,v3] |
|-------|-------------------|---------|----------------------|
| 0 | 33.9 | 1.000x | [5, 5, 3, 3] |
| 1 | 33.9 | 1.000x | [5, 5, 5, 5] |
| 2 | 33.9 | 1.000x | [5, 5, 5, 5] |
| 3 | 33.9 | 1.000x | [5, 5, 5, 5] |
| 4 | 33.9 | 1.000x | [5, 5, 5, 5] |
| 5 | 33.9 | 1.000x | [5, 5, 5, 5] |
| 6 | 33.9 | 1.000x | [5, 5, 5, 5] |
| 7 | 33.9 | 1.000x | [5, 5, 5, 5] |
| 8 | 33.9 | 1.000x | [5, 5, 5, 5] |
| 9 | 33.9 | 1.000x | [5, 5, 5, 5] |
| 10 | 33.9 | 1.000x | [5, 5, 5, 5] |
| 11 | 33.9 | 1.000x | [5, 5, 5, 5] |
| 12 | 33.9 | 1.000x | [5, 5, 5, 5] |
| 13 | 33.9 | 1.000x | [5, 5, 5, 5] |
| 14 | 33.9 | 1.000x | [5, 5, 5, 5] |
| 15 | 33.9 | 1.000x | [5, 5, 5, 5] |
| 16 | 33.9 | 1.000x | [5, 5, 5, 5] |
| 17 | 33.9 | 1.000x | [5, 5, 5, 5] |
| 18 | 33.9 | 1.000x | [5, 5, 5, 5] |
| 19 | 33.9 | 1.000x | [5, 5, 5, 5] |

### 3.3 MAP-Elites 数据库状态

| 指标 | 值 |
|------|-----|
| Elite 更新次数 | 4次 (整个运行) |
| 最佳 score | 8.0 (perf_score=5 + 3×1.0=8.0) |
| 占据的 cells | (2,3,3,0), (3,3,3,0) |
| Archive 覆盖率 | 2/256 = 0.78% |

---

## 4) 性能停滞根因分析

### 4.1 直接原因: DPAS 指令使用方式错误

**v4 成功模式 (11.4ms, 2.98x speedup):**
```c
// 正确: 8×16 tile per subgroup, 全部 float8 元素都使用
float8 acc = 0.0f;
short8 a_val;  // A: packed as short8
int8 b_val;    // B: packed as int8 (pairs of half)
acc = intel_sub_group_f16_f16_matrix_mad_k16(a_val, b_val, acc);
// 输出: 所有 8 elements 写回 C
for (int r = 0; r < 8; r++) {
    C[row_idx * N + col_idx] = convert_half(((float*)&acc)[r]);
}
```

**v5 失败模式 (33.9ms, 1.0x, Trial 0):**
```c
// 错误: 虽然调用了 DPAS 但只取 s0
half16 a_vec = vload16(0, &Asub[ly][0]);
half16 b_vec; // 错误类型! 应该是 int8
float8 dpas_acc = (float8)(0.0f);
dpas_acc = intel_sub_group_f16_f16_matrix_mad_k16(a_vec, b_vec, dpas_acc);
acc += dpas_acc.s0;  // ← 只取第一个元素，浪费 87.5% 计算
```

**关键差异:**
1. **A 操作数类型**: v4用`short8`(正确的DPAS格式), v5用`half16`(通用SIMD格式)
2. **B 操作数类型**: v4用`int8`(两个half打包为int), v5用`half16`(错误)
3. **结果利用**: v4使用全部8个float结果, v5多数情况只用`s0`
4. **Tile 映射**: v4是8×16 per subgroup (匹配DPAS硬件), v5是16×16 (不匹配)

### 4.2 间接原因: Host 端 Launch 参数不匹配

`task.py` 中的 launch 逻辑:
```python
reqd = re.search(r"reqd_work_group_size\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)", src_text)
if reqd:
    lws_x, lws_y = int(reqd.group(1)), int(reqd.group(2))
    gws_x = int(math.ceil(n_int / lws_x)) * lws_x
    gws_y = int(math.ceil(m_int / lws_y)) * lws_y
```

**v4 kernel (无 reqd_work_group_size)**:
- 走 fallback 路径: `knl(queue, (N, M), None, ...)`
- Runtime 自动选择合适 LWS，subgroup=16
- GWS=(2048, 2048), 每个 subgroup 处理 8×16 tile → 有效

**v5 kernel (有 reqd_work_group_size(16,1,1))**:
- 走 parsed 路径: `LWS=(16,1)`, `GWS=(2048, 2048)`
- 产生 2048 个 Y-direction work-groups
- 但 kernel 每组处理 16 行, 所以只需 ceil(2048/16)=128 个 Y 组
- **1920 个 work-groups 空跑** (93.75% 浪费)

**v5 kernel (有 reqd_work_group_size(16,16,1))**:
- `LWS=(16,16)`, `GWS=(2048, 2048)`
- 产生 128×128=16384 work-groups
- 但 16×16 work-items 做 SLM 协作加载 → 标量逐元素累加
- **等同于 naive tiling 而非 DPAS 矩阵计算**

### 4.3 根本原因: 模型能力差异

| 模型 | DPAS 理解能力 | 生成质量 |
|------|--------------|----------|
| claude-4-6-opus (temp=0.0, v4) | 精确理解 `short8/int8` 操作数格式 | 一次命中最优 |
| claude-4-6-opus (temp=0.3, v5) | 温度引入随机性，偏离最优解 | 生成正确但非最优代码 |
| claude-4-5-sonnet (temp=0.6, v5) | 理解概念但格式细节错误 | 用 half16 替代 short8/int8 |
| gpt-5.3-codex (temp=0.4, v5) | 知道 DPAS 存在但不了解精确语义 | 调用了 DPAS 但只取 s0 |

**核心矛盾: DPAS 编程是极度精确的硬件级优化，对代码格式的要求极高（操作数打包方式、tile 大小、子组映射必须精确匹配）。这类任务需要最强模型在最确定性的设置下工作，而非多模型集成+高随机性。**

---

## 5) 进化算法行为分析

### 5.1 优化目标 Profile 分布

| Profile | 出现次数 | 策略 |
|---------|----------|------|
| memory=3, compute=3, parallelism=3, esimd=0 | 17 | mutate |
| memory=3, compute=0, parallelism=1, esimd=0 | 10 | mutate |
| memory=2, compute=3, parallelism=3, esimd=0 | 8 | mutate |
| memory=3, compute=1, parallelism=3, esimd=0 | 7 | mutate |
| memory=2, compute=0, parallelism=2, esimd=0 | 7 | mutate |
| 其他 (mutate) | 26 | mutate |
| 其他 (diversify) | 5 | diversify |

**问题**: 
- esimd=0 占 75/80 (93.75%), 几乎未探索高级 ESIMD/DPAS 空间
- mutate 策略 75/80 (93.75%), diversify 仅 5/80 
- `exploration_ratio=0.4` 的效果有限——因为初始种群只有 1 个 program0

### 5.2 岛模型与迁移

- 仅 4 次 elite 更新（整个 80 evaluations）
- 2 个 cells 被占据: (2,3,3,0) 和 (3,3,3,0)
- 未触发有效迁移（archive 太稀疏）

### 5.3 推理时间统计

| 模型 | 调用次数 | 平均耗时 | 最快 | 最慢 |
|------|----------|----------|------|------|
| claude-4-6-opus | 30 | 45.5s | 29.7s | 99.4s |
| claude-4-5-sonnet | 29 | 62.2s | 45.2s | 82.2s |
| gpt-5.3-codex | 21 | 20.7s | 15.1s | 27.5s |

---

## 6) v4 成功 vs v5 失败的决定性因素

### 6.1 对比总结

| 维度 | v4 (成功, 2.98x) | v5 (失败, 1.0x) |
|------|-------------------|------------------|
| 模型 | claude-4-6-opus only | 3 模型混合 |
| 温度 | 0.0 (确定性) | 0.3-0.6 (随机性) |
| 迭代 | 4 trials × 2 branches = 8 | 20 trials × 4 branches = 80 |
| 首轮成功率 | Trial 0 即获 2.93x | Trial 0 最佳仅 1.0x |
| DPAS 操作数 | short8 + int8 (正确) | half16 + half16 (错误) |
| Tile 大小 | 8×16 (匹配 DPAS) | 16×16 (不匹配) |
| reqd_work_group_size | 无 (runtime 选择) | 有 (但与 host launch 不匹配) |

### 6.2 核心教训

**对于这个特定任务 (Intel DPAS matmul):**

1. **模型质量 > 模型数量**: claude-4-6-opus 在 temperature=0 时对 DPAS intrinsics 的理解远超其他模型。引入 gpt-5.3-codex 和 claude-4-5-sonnet 不是增加多样性，而是引入噪声。

2. **精确性 > 随机性**: DPAS 编程需要精确的类型匹配(short8/int8)和 tile 大小(8x16)。temperature > 0 导致模型"创新性地"使用 half16，看似合理但性能全失。

3. **简单参数调优不足以改善此类问题**: 当进化的搜索空间中只存在一个很窄的最优区域(正确的 DPAS 用法)时，增加搜索广度但降低精度会导致覆盖面更大却永远找不到针尖。

4. **Host launch 配置是隐性约束**: 生成的 kernel 必须与 `task.py` 的 launch 逻辑配合。v4 的无 `reqd_work_group_size` 恰好让 runtime 选择最优 LWS。

---

## 7) 改进方案修订

基于 v5 实验的教训，对 v2 文档中的建议重新评估：

### 7.1 已验证有效的方向

| 建议 | 状态 | 修订 |
|------|------|------|
| A. 增大 max_iters/branches | ✓ 实施了 | 有帮助但需配合模型质量 |
| B. 提高 exploration_ratio | ✓ 实施了 | 效果有限（种群太小时探索空间太大） |

### 7.2 已验证有害的方向

| 建议 | 状态 | 原因 |
|------|------|------|
| D. 增加 LLM temperature | **有害** | DPAS 编程需要精确性，温度破坏了正确的类型选择 |
| L. 多模型协同进化 | **有害** | 弱模型无法理解硬件 intrinsics，产出大量 1.0x 噪声 |

### 7.3 修订后的优先级建议

| 优先级 | 方案 | 预期收益 | 关键要求 |
|--------|------|----------|----------|
| **P0** | 恢复 claude-4-6-opus 单模型 + temp=0 | 恢复 2.98x 基础 | 配置回退 |
| **P0** | Warm-start: 从 v4 最优 kernel 开始进化 | 在 2.98x 基础上继续提升 | 设置 kernels_iter_0_path |
| **P1** | 修复 reqd_work_group_size 与 host launch 匹配 | 避免空 work-group 浪费 | 修改 task.py 或 prompt |
| **P1** | 在 prompt 中注入正确的 DPAS 操作数格式示例 | 确保类型正确 | 修改 USER_INSTRUCTIONS |
| **P2** | 多模型只用强模型: claude-4-6-opus × N (不同seed) | 保持质量+增加多样性 | 修改 ensemble config |
| **P2** | Tabu list for target profiles | 消除重复 | 代码改动 |
| **P3** | 仅在正确 kernel 进入 archive 后启用 temp>0 | 在已证明的解附近探索 | 分阶段策略 |
| **P3** | 更细粒度的 Fitness shaping | 区分 "调用了DPAS但用错" vs "正确使用DPAS" | Fitness 改动 |

### 7.4 推荐的下一步配置

```yaml
# 最优配置建议 (基于 v4+v5 对比)
inference:
  servers:
  - _target_: kernelgen.inference_server.InferenceServer
    server_type: intel_gnai
    model_name: claude-4-6-opus    # 最强模型独占
    temperature: 0.0               # 确定性模式
    max_tokens: 10000

max_iters: 20
branches_per_iteration: 4

# 从 v4 最优解开始
kernels_iter_0_path: "runs/b580_matmul_dpas_v4"

# 进化参数保持平衡
database:
  config:
    exploration_ratio: 0.3
    exploitation_ratio: 0.6

# 确保 prompt 中包含正确的 DPAS 类型信息
prompt:
  include_hardware_specs: true
  num_optimization_tips: 3
```

同时应在 `USER_INSTRUCTIONS` 中添加:
```
CRITICAL: The intel_sub_group_f16_f16_matrix_mad_k16 intrinsic requires:
- A operand: short8 (NOT half16) - represents 8 rows packed as short
- B operand: int8 (NOT half16) - represents 16x16 tile with pairs of half packed as int
- Result: float8 - represents 8 row results
- Tile size: 8 rows × 16 cols per subgroup
- Do NOT use reqd_work_group_size attribute - let runtime choose optimal LWS
```

---

## 8) 时间分解

| 阶段 | 平均时间/Trial | 占比 |
|------|----------------|------|
| LLM 推理 (4 branches 并行) | ~65s (受最慢模型制约) | 37% |
| 评估 (4 branches 串行 GPU) | ~100s | 57% |
| 采样 + 数据库操作 | ~10s | 6% |
| **总计/Trial** | **~175s** | 100% |
| **总计 20 Trials** | **3508.5s (58.5 min)** | — |

---

## 9) 关键发现总结

1. **多模型集成对高精度硬件优化任务有害**: DPAS 编程是一个对类型和格式极度敏感的任务。弱模型无法理解 `short8/int8` 的打包语义，生成的代码表面正确（能编译、通过正确性测试）但性能无提升。

2. **Temperature 对 DPAS 有害**: 即使 claude-4-6-opus 在 temperature=0.3 时也比 temperature=0 时差，因为它会"创造性地"选择 half16 替代 short8。

3. **80 个样本全部锁定在 33.9ms**: 说明所有生成的 kernel 都退化为等效于参考的标量实现（DPAS 指令虽被调用但输出未被有效利用）。

4. **进化算法本身工作正常**: 采样、变异、评估、数据库更新链路均正常。问题出在 LLM 输出质量。

5. **配置调优的教训**: "增加搜索广度+多样性" 的通用优化策略在此任务上失败，因为目标函数的最优解是一个极度精确的代码模式，宽泛搜索反而远离了它。

---

## 10) 与 v4 运行数据的全面对比

| 指标 | v4 | v5 | 变化 |
|------|-----|-----|------|
| 总 kernel 评估 | 8 | 80 | +10x |
| 最佳 speedup | 2.98x | 1.0x | **-66%** |
| 正确性率 | 7/8 (87.5%) | 76/80 (95%) | +7.5% |
| 编译率 | 8/8 (100%) | 80/80 (100%) | = |
| 总耗时 | 269.6s | 3508.5s | +13x |
| 使用模型数 | 1 | 3 | +2 |
| LLM temperature | 0.0 | 0.3-0.6 | +0.3-0.6 |
| 最佳得分 (combined) | 13.95 | 8.0 | **-42%** |
| DPAS 类型正确率 | 高 (short8/int8) | 低 (half16/half16) | ↓ |

---

## 11) 产物文件索引

| 类型 | 路径 |
|------|------|
| 运行目录 | `runs/b580_matmul_dpas_v5/` |
| 总结结果 | `runs/b580_matmul_dpas_v5/results.json` |
| 控制器日志 | `runs/b580_matmul_dpas_v5/controller.log` |
| 评估结果 | `runs/b580_matmul_dpas_v5/eval_result_*_trial_*_v*.json` (80个) |
| 生成代码 | `runs/b580_matmul_dpas_v5/generated_kernel_*_trial_*.py` (20个best) |
| Prompt记录 | `runs/b580_matmul_dpas_v5/prompt_*_trial_*_v*.md` (80个) |
| 任务配置 | `workspace/ocl_matmul_dpas/config.yaml` |
| v4对比目录 | `runs/b580_matmul_dpas_v4/` |
