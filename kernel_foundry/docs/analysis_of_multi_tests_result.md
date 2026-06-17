# KernelFoundry 多次相同命令实验结果分析

**硬件**:Intel Arc B580 (Battlemage G21, device_id `0xe20b`, XMX/DPAS)
**任务**:OCL matmul fp16,shape (M,K,N) = (2048, 2560, 2048),≈21.5 GFLOP
**参考基线**:reference kernel ≈ **33.9 ms**(六次测得一致,稳定)
**日期**:2026-06-15
**代码版本**:git commit `98f26eb`,oneAPI 2025.0,icpx 2025.0.4,NEO 25.48.36300.8,torch 2.9.0+xpu

---

## 1. 实验设计

用**完全相同的命令**(仅 `job_name` / `task_name` 后缀不同以隔离 DB)连续运行 6 次:

```
python scripts/run_custom_task.py custom_task=.../ocl_matmul_dpas \
  task_origin=local_gemm gpu_arch=bmg language=OCL \
  use_queue=false use_container=false store_generated_kernels_in_db=false \
  validate=true max_iters=6 branches_per_iteration=4 \
  stop_once_correct=false has_reference_build_step=false \
  eval_config.profile_custom_model=false eval_config.profile_original_model=false \
  test_timeout=180 build_timeout=120
```

> 注:第一次 `b580_matmul_dpas` 用了 `max_iters=12`,后四次用 `max_iters=6`。但所有运行的分叉都在 trial 0/1 就已发生,iters 数不是差异主因。

---

## 2. 六次结果对照

| 运行 | 最佳 runtime | **加速比** | 算力 | 正确率 | 命中高效解 | 落入峰 | 最佳候选 |
|---|---|---|---|---|---|---|---|
| `b580_matmul_dpas` | 33.9 ms | **1.00×** | ~0.6 TFLOPS | 45/48 | **从未** | 低峰 | trial_5_v2 |
| `b580_matmul_dpas_v1` | 13.5 ms | **2.52×** | ~1.6 TFLOPS | 18/24 | trial 1 | 中 | trial_5_v2 |
| `b580_matmul_dpas_v2` | 1.05 ms | **32.3×** | ~20.5 TFLOPS | 17/24 | trial 1 | 高峰 | trial_4_v2 |
| `b580_matmul_dpas_v3` | 1.01 ms | **33.6×** | ~21.3 TFLOPS | 16/23 | trial 0 | 高峰 | trial_3_v2 |
| `b580_matmul_dpas_v4` | 33.9 ms | **1.00×** | ~0.6 TFLOPS | 7/24 | **从未** | 低峰 | trial_4_v1 |
| `b580_matmul_dpas_v5` | **0.623 ms** | **54.4×** | **~34.5 TFLOPS** | 21/24 | **trial 0** | **高峰(最佳)** | trial_1_v1 |

**同一条命令,产出了 1.00× / 2.52× / 32.3× / 33.6× / 1.00× / 54.4× 六种结果。**

**第五次(v4)是关键对照**:它打破了"后几次都高"的表象,**重新落回低性能峰(1.0×)**。这彻底证伪了"逐次累积递增"的泄漏假说——若真有累积,v4 不可能跌回 1.0×。

**第六次(v5)是另一个关键对照**,且**两次否定累积假说**:
1. v5 紧跟在跌回低峰的 v4 之后,却**直接冲到全局最佳 54.4×**——序列从 1.0×(v4)跳到 54.4×(v5),再次证明每次运行彼此独立、不存在"接力变好"。
2. v5 **首次刷新了高峰上限**:0.623 ms ≈ 34.5 TFLOPS,**显著快于 v2/v3 的 ~1.0 ms / ~21 TFLOPS**。这推翻了之前"~21 TFLOPS 是物理上限"的推断——高峰本身是一个**有宽度的区间(~21–35 TFLOPS)**,而非单一收敛点;真正上限更高,只是采样很少触及。
3. v5 **6 个 trial 全部落入高峰**(24 个候选中 21 个正确,无一退化到 1.0× 以外的低峰),是六次里唯一"满堂红"的运行——与 v4"开局全错"形成最极端的对比。

六次序列 **{1.0×, 2.5×, 32×, 34×, 1.0×, 54×}** 无序剧烈波动,完全符合"每次独立随机抽样、落入双峰之一"的预期。

### v4 的独特轨迹(进一步佐证随机性)

- **trial 0/1/2 全 12 个候选 correctness 失败**(全 -1.0):连续三轮 LLM 采样都没产出可用 kernel,这是五次里独有的"冷开局";
- trial 2 那批坏 kernel 触发 **每个 correctness 测试跑满 ~100 秒**(正常约 11 秒),说明 kernel 内含病态循环 / 错误 launch;
- 运行中还出现一次 **Bedrock `InternalServerException`**(自动重试成功)——后端偶发错误也是不可控因素之一;
- 直到 trial 3-5 才出现 correct 候选,但全部停在 33.9-34.0 ms(低峰),没能逃出。

### v5 的独特轨迹(全程高峰,且仍在进化中上升)

- **trial 0 开局即破峰**:v0/v1 correctness 失败,但 v3 立刻拿到 1.11 ms / 30.5×,首个 trial 就锁定高峰精英;
- **逐 trial 稳步精炼**(典型的 MAP-Elites 围绕高峰精英变异):各 trial 内最佳 runtime 大致 1.11 → 0.623 → 0.749 → 0.905 → 0.736 → 0.636 ms,全部落在 0.6–1.1 ms 高峰区间;
- **全局最佳 trial_1_v1 = 0.623 ms / 54.4×**,是六次里最快的单个候选;
- **24 个候选中 21 个正确**,仅 3 个 correctness 失败(trial_0_v0/v1、trial_1_v2),命中率为六次最高;
- 全程无 Bedrock 报错、无病态超时,运行时间也最短(约 7 分钟),与 v4 形成两极对照。

### 最佳候选真实性核验(均已验证为真实测量,非测量假象)

| 运行 | runtime | std | 测量次数 | correctness | 是否 fallback |
|---|---|---|---|---|---|
| b580 | 33.9 ms | 0.023 ms | 29 | True(但=baseline) | 否 |
| v1 | 13.5 ms | 0.035 ms | 74 | True | 否 |
| v2 | 1.05 ms | 0.023 ms | 235 | True | 否 |
| v3 | 1.01 ms | 0.0025 ms | 140 | True | 否 |
| v4 | 33.9 ms | 0.127 ms | 29 | True(但=baseline) | 否 |
| v5 | 0.623 ms | 0.00496 ms | 700 | True | 否 |

- 所有"快"结果都通过完整 pytest correctness(wrt_pytorch + wrt_reference);
- runtime_stats 是数百次重复测量的统计,std < 2%,不是 -1.0 sentinel 或 0.0 占位;
- 日志中 `matmul_reference` / `fallback launch` 匹配均为良性(失败候选的 launch 警告 + 测试夹具加载基线源码),**无候选退化到参考实现**。

---

## 3. 核心问题排查:"后面是否参考了前面的结果?"

由于前三次结果呈单调递增(1.0× → 2.5× → 32×),曾怀疑存在跨运行状态泄漏(后面的运行读到了前面找到的好 kernel)。**系统排查 + 第五、六次对照后全部排除:**

| 潜在泄漏通道 | 检查结果 | 结论 |
|---|---|---|
| `/tmp` build/test 临时目录 | tempfile 机制运行后自动删除,无残留 | ✗ 无泄漏 |
| `/tmp` 持久化 db/checkpoint/archive | 仅有 6月11日另一任务(flash-attention)的文件,与 matmul 无关 | ✗ 无关 |
| `runs/kernels.sqlite3` 数据库 | 仅存 baseline 行,**全是 33.9–34.0 ms / score=5**,无任何 13.5/1.05 ms 的快 kernel 记录 | ✗ 无快 kernel 可参考 |
| MAP-Elites 存档继承 | 六次启动日志全部 `Total programs: 0 / Available for evolution: 0`,均从空存档冷启动 | ✗ 不继承 |
| DB 灵感检索隔离 | 代码按 `task_name` 过滤(`evolve_database_optimization_aware.py:1521`, `:1590`),六次 task_name 各异,天然隔离 | ✗ 互不可见 |

**决定性证据**:即便存在跨运行读取,DB 里也只存了 33.9 ms 的 baseline——**根本不存在 v1 的 13.5 ms 或 v2/v3/v5 的亚毫秒成果供后续参考**。

**三个独立证伪**:
1. **v3(第四次)否定"累积递增"**:v3 加速比 33.6× ≈ v2 的 32.3×,没有继续显著上升。
2. **v4(第五次)否定"逐次变好"**:v4 **跌回 1.00×**(低性能峰)。若存在任何形式的跨运行累积 / 参考,v4 绝不可能比 v2/v3 差 30 倍。
3. **v5(第六次)再次否定 + 反向证伪**:v5 紧接 v4 之后,却从 1.0× **暴涨到 54.4×**(全局最佳),且**反超** v2/v3。若结果按运行顺序"接力"或受前次影响,既无法解释 v4→v5 的暴涨,也无法解释 v5 超越更早的 v2/v3。六次序列 **{1.0×, 2.5×, 32×, 34×, 1.0×, 54×} 无序剧烈波动**,正是"每次独立随机抽样、落入双峰之一"的铁证。

---

## 4. 巨大差异的根本原因

差异 = **双层随机性 × 进化取极值 × 双峰解空间**。

### 4.1 双层随机性(两个独立的不可复现来源)

| 层 | 随机源 | 是否受 `random_seed=42` 控制 | 后果 |
|---|---|---|---|
| **① Prompt 构造** | `np.random.choice(KERNEL_OPTIMIZATION_TIPS, n_tips, replace=False)`(`template_manager.py:183`) 随机抽取注入哪些优化技巧 | 名义受控,但随调用次数 / task_name **漂移**,跨运行不能稳定复现 | 不同运行喂给 LLM 的"提示"不同 |
| **② LLM 采样** | Bedrock `invoke_model`,temperature=0.1,**请求体无 `seed` 字段**(`gnai_inference.py:411-425`) | **完全不受控** | 即便 prompt 相同,也产出不同 kernel |

**实证**:
- 前三次 trial_0 prompt md5 相同(`a19bd733`, 16672 B),但生成的 kernel md5 **全不同** → 证明 ② LLM 采样非确定(相同输入 → 不同输出)。
- trial_0 prompt md5 在六次里出现 **三个不同值**:`a19bd733`(b580/v1/v2)、`f20d0784`(v3/v4)、`823e7ffa`(v5, 16695 B)。v5 注入的技巧又换了一组(Aggressive B prefetch + SLM bank-conflict striding + cooperative/three-level prefetch + double/triple buffering),与前两组都不同 → 进一步坐实 ① prompt 构造本身带随机性。
- **六次 trial_0 生成 kernel md5 全不同**:
  - b580: `f89ce718` (11903 B)
  - v1: `4a162a5c` (21639 B)
  - v2: `ce71a0be` (15586 B)
  - v3: `893fb11f` (15178 B)
  - v4: `5b8bfa69` (12974 B)
  - v5: `b9ba2e6e` (15477 B)

> `random_seed=42`(`evolve_database_optimization_aware.py:741-744`)只 seed 框架侧的 Python/numpy RNG(parent 选择、维度采样),**既管不到 LLM,也无法在跨运行 / 跨 task_name 下稳定固定 prompt tip 抽样**。

### 4.2 进化算法把"采样运气"放大为"最终性能"

MAP-Elites 对结果分布是**取极值(max)而非取平均**。每次运行 = 在解空间抽 6×4 = 24 个样本,取最好的并围绕它精炼。一旦某 trial 命中高效解,精英保留机制立即锁定该 cell 并持续变异精炼;反之若初始精英是"正确但慢"的局部最优,整个种群被其基因锁死(b580 / v4 即如此,全程卡 33.9 ms)。

**v4 与 v5 是这套放大机制的两个极端**:
- **v4(放大坏开局)**:trial 0-2 连 correct 都没抽到,半数 trial 被浪费,种群始终没逃出低峰 → 1.0×。
- **v5(放大好开局)**:trial 0 就抽中高峰精英(v3=1.11 ms),后续 trial 围绕它持续变异精炼,逐步压到 0.623 ms,24 个候选 21 个正确 → 54.4×。
**同一套算法,开局采样的好坏被进化机制成倍放大成天差地别的最终性能**——这正是"差异巨大"的核心放大器。

### 4.3 matmul 解空间是"双峰"的

- **低性能峰**:"形式上正确的 DPAS,但访存 / 数据复用模式退化" → 带宽受限,1~2.5×(b580, v4 落 1.0×;v1 略偏中 2.5×)。
- **高性能峰**:"正确且真正打满 XMX(正确 SLM 布局 / sub-group 数据流 / K-tile 流水 / 多级 prefetch)" → **~21–35×,且本身是一个有宽度的区间**(v2/v3 落在 ~21 TFLOPS 一侧,v5 冲到 ~34.5 TFLOPS)。

写出"形式正确的 DPAS"很容易,但落在哪个峰取决于采样运气。这解释了为什么差异不是连续的 1.5×/2× 渐变,而是**伯努利式的"命中 / 未命中"双峰**:六次中两次落低峰(b580/v4)、一次居中(v1, 2.5×)、三次落高峰(v2/v3/v5)。粗略估计单次命中高性能峰的概率约 **3/6 ≈ 50%**(样本量小,仅作量级参考)。**注意高峰内部也有显著方差**:即便都命中高峰,v5(0.623 ms)仍比 v2/v3(~1.0 ms)快约 40%,说明"命中高峰"之后还有第二层"在高峰内能精炼到多好"的随机性。

### 4.4 高峰是一个区间,存在硬件上限但远未触顶

最初基于 v2/v3(都约 ~1.0 ms / ~21 TFLOPS)推断"~21 TFLOPS 是物理上限"。**v5 推翻了这个推断**:它做到 0.623 ms ≈ **34.5 TFLOPS**,比 v2/v3 快约 40%。可见:

- 高峰不是单一收敛点,而是 **~21–35 TFLOPS 的区间**;之前的"上限"只是两次样本恰好落在区间下沿造成的错觉;
- 对 B580(fp16 XMX 理论峰值约 230 TFLOPS)而言,即便 v5 的 34.5 TFLOPS 也只是理论峰值的 ~15%,**真正的硬件上限远未触及**,说明高峰内仍有很大优化空间没被采样稳定命中;
- 但"存在上限、非无限增长"的结论依旧成立:命中高效解后逼近(而非超越)硬件能力,六次都没出现脱离物理合理范围的数值,从机制上排除"无限递增的泄漏"。v4 跌回 1.0× 也从反面印证:性能不会跨运行单调累积。

---

## 5. 结论与方法学建议

1. **该流程本质非确定、不可复现**。"同命令不同结果"是设计使然(双层随机性 + 无 LLM seed),**不是 bug,也不是跨运行泄漏**。

2. **绝非"后面参考前面"**。六次均从空存档冷启动,DB 仅存 baseline,无泄漏通道;v3 ≈ v2 不再上升、v4 跌回 1.0×、v5 紧接 v4 又暴涨到 54× 且反超更早的 v2/v3——三个独立证据彻底否定累积假说。

3. **单次运行结果不可信**。同配置可落在 1.0×~54× 任意点。六次实测分布 **{1.0×, 2.5×, 32×, 34×, 1.0×, 54×}**,median ≈ 18×(介于 2.5× 与 32× 之间),但呈明显双峰(低峰 ~1× 两次、中段 2.5× 一次、高峰 ~21–54× 三次)。**只跑一次拿到 54× 或 1× 都不能代表该方法的真实能力**;且即便都命中高峰,最终性能仍可相差 40%(v5 vs v2/v3),所以连"高峰内的数字"也需多跑取分布。

4. **实践建议**(若需可信 / 可复现结论):
   - **多次运行取分布**(至少 3–5 次,报告 min / median / max 及双峰占比);本次 6 次已显示方差极大(1.0×~54×),生产评估应跑更多;
   - **固定 LLM `seed` + 锁定 temperature**(需改 `gnai_inference.py` 在 invoke_model body 中传 seed);
   - **锁定 prompt tips**(固定 `template_manager.py` 的 tip 抽样,或显式传入完整 tip 集);
   - **用已验证的高效 kernel 作种子**入库,避免每次冷启动"重抽彩票",直接从高性能峰开始精炼;
   - **增大 `branches_per_iteration` / `max_iters`** 提高每次运行命中高性能峰的概率(更多采样 = 更可能抽中长尾高效解),代价是更多 Bedrock 调用与时间。

---

## 附:关键代码位置

- LLM 调用无 seed:`kernelgen/gnai_inference.py:411-425`(`BedrockInferenceServer.__call__`,body 仅含 messages/max_tokens/anthropic_version/temperature/top_p)
- 框架 RNG seed:`kernelfoundry/kernelfoundry/algorithm/evolve_database_optimization_aware.py:741-744`
- Prompt tip 随机抽样:`kernelfoundry/kernelfoundry/algorithm/prompts/template_manager.py:183`
- DB 按 task_name 隔离检索:`kernelfoundry/kernelfoundry/algorithm/evolve_database_optimization_aware.py:1521, 1590`
- kernels DB 默认路径:`kernelfoundry/configs/paths/default.yaml`(`kernels_db_path: sqlite:///runs/kernels.sqlite3`,readonly/insertonly 默认 null)
