# KernelFoundry 内核搜索优化方案

**目标硬件**:Intel Arc B580 (Battlemage G21, XMX/DPAS)
**基准任务**:OCL matmul fp16 (2048×2560×2048)
**依据**:六次相同命令实验(见 `analysis_of_multi_tests_result.md`)+ 早先调优经验
**代码版本**:git commit `98f26eb`

---

## 0. 出发点:六次实验暴露的三个核心痛点

| 现象(实测) | 根因 | 本方案对应改进 |
|---|---|---|
| 同命令六次结果 {1.0×, 2.5×, 32×, 34×, 1.0×, 54×},命中高峰仅 ~50% | 双层随机性 + 进化取极值 + 双峰解空间;**没有任何提前结束/早停机制**,跑满 `max_iters` 才停 | §1 早停 + §4 多次运行取分布 |
| v4 三个 trial(12 候选)全 correctness 失败,每个坏 kernel 把 correctness 测试拖到 ~100 s;高峰候选却要跑满 100 次 perf trial | 评估成本固定:`num_perf_trials=100`、完整 profiling 对**好坏候选一视同仁**,坏候选浪费大量预算 | §2 粗→精分级评估 |
| trial_0 prompt 出现 3 个不同 md5;prompt 只随机塞 **2 条** 通用 tip,**无 XMX/DPAS 专门指导**,默认 `use_feedback_llm=False` | prompt 构造随机且信息量低;反馈回路弱;缺硬件先验 | §3 更有效的 prompt |

**总目标**:在**相同或更低的 Bedrock 调用预算**下,把"命中高性能峰"的概率从 ~50% 提到尽可能高,并让单次运行更快收敛、更可复现。

---

## 1. 搜索提前结束机制(Early Stopping)

### 1.1 现状

[controller.py:615](kernelfoundry.internal/kernelfoundry/kernelfoundry/algorithm/controller.py#L615) 的主循环 `for trial in range(config.max_iters)` **唯一的提前退出**是 [controller.py:798-799](kernelfoundry.internal/kernelfoundry/kernelfoundry/algorithm/controller.py#L798):

```python
if config.stop_once_correct and kernel_exec_result.compiled and kernel_exec_result.correctness:
    break
```

问题:`stop_once_correct=true` 太激进(第一个**正确**候选就停,通常落在低峰 1.0×),`false` 又跑满所有 iter。**两个极端之间没有任何"性能已收敛就停"或"明显没救了就停"的中间策略**。

### 1.2 方案:三类早停条件(在 L798 旁并列插入)

在 `controller.py:798` 同一位置,新增一个 `should_stop_early(trial, database, config)` 调用:

```python
# 既有
if config.stop_once_correct and kernel_exec_result.compiled and kernel_exec_result.correctness:
    break
# 新增
stop, reason = should_stop_early(trial, self.program_database, best_history, config)
if stop:
    logging.info(f"[early-stop] trial {trial}: {reason}")
    break
```

三类条件(取并集,任一满足即停):

**(A) 性能平台早停(patience)** —— 解决"已经收敛还在烧钱"
- 维护 `best_history`(每 trial 结束时的全局最佳 `runtime_improvement`)。
- 若连续 `early_stop_patience`(建议 **3**)个 trial,最佳值相对提升 < `early_stop_min_delta`(建议 **2%**),则停。
- 针对实测:v5 在 trial 1 就到 0.623 ms,后面 4 个 trial 只在 0.6–0.9 ms 抖动 → 本可在 trial 3 左右停,省 ~40% 预算。

**(B) 目标达成早停(target speedup)** —— 解决"命中高峰后无谓精炼"
- 若全局最佳 `runtime_improvement >= early_stop_target_speedup`(B580 建议 **30×**,即已稳进高峰),再多跑 `target_patience`(建议 **1**)个 trial 确认不退化后停。
- 区别于 `stop_once_correct`:它要求的是**高性能**而非仅**正确**,避免锁死在低峰。

**(C) 绝望早停(no-hope)** —— 解决 v4 式"开局全废还硬跑满"
- 若前 `nohope_warmup`(建议 **3**)个 trial **从未产出任何 correct 候选**,或最佳仍 `< 1.5×`,判定本次冷启动运气太差。
- 动作二选一(配置 `nohope_action`):
  - `stop`:直接放弃本次,交给上层"多次运行"策略重抽(推荐,配合 §4);
  - `reseed`:强制下一 trial 用"已验证高峰 kernel"作种子(配合 §3.4),把一次烂运行救回来。

### 1.3 新增配置(`run.yaml`)

```yaml
early_stop:
  enable: true
  patience: 3              # (A) 连续无显著提升的 trial 数
  min_delta: 0.02          # (A) 显著提升阈值(相对 2%)
  target_speedup: 30.0     # (B) 达到即准备停(B580 高峰下沿 ~21×,留余量取 30)
  target_patience: 1       # (B) 达标后再确认的 trial 数
  nohope_warmup: 3         # (C) 判定绝望的观察窗口
  nohope_min_speedup: 1.5  # (C) 窗口内最佳低于此值视为绝望
  nohope_action: stop      # (C) stop | reseed
```

### 1.4 预期收益
- 高峰运行(v2/v3/v5 类)平均省 **30–40%** trial;
- 绝望运行(v4 类)早早止损或被 reseed 救回;
- 配合 §4,在固定总预算下能跑更多次独立运行 → 命中高峰的总概率↑。

---

## 2. 先粗略搜索,再精确搜索(Coarse-to-Fine)

### 2.1 现状:评估成本对所有候选一视同仁

每个候选都按 [run.yaml:96-102](kernelfoundry.internal/configs/run.yaml#L96) 的满配评估:
- `num_perf_trials: 100`、`warmup_min_iters: 10`、`warmup_min_time: 0.1`、`inner_loop_min_time: 0.01`;
- `profile_custom_model: true`(完整 profiling)。

实测代价:v5 单个高峰候选要跑 **600–1200 次** perf 测量;v4 的坏候选还会把 correctness 测试拖到 ~100 s。**坏候选和好候选花一样的钱**,极其浪费。

性能测量的 warmup/trials 是**自适应且 config 驱动**的([performance.py:251-276](kernelfoundry.internal/kernelfoundry/kernelfoundry/eval_pipeline/utils/performance.py#L251)),所以**调小这些值就能得到一个廉价的粗测**——这是分级评估的关键支点。

### 2.2 方案:两段式评估(粗筛 → 精测)

把"一个候选 = 一次满配评估"改为**两阶段漏斗**:

```
branches_per_iteration 个候选
        │
   ┌────▼─────────────────────────┐
   │ 粗筛 (coarse):                │  廉价、只为排序
   │  · correctness 必过(短 timeout)│
   │  · 少量 perf trial 估个量级    │
   │  · 不做 profiling             │
   └────┬─────────────────────────┘
        │ 取 top-K(按粗测 runtime)
   ┌────▼─────────────────────────┐
   │ 精测 (fine):                  │  昂贵、只给赢家
   │  · 满配 num_perf_trials=100   │
   │  · 完整 profiling             │
   └──────────────────────────────┘
```

**粗筛配置**(临时 override eval_config,约省 90–95% 测量时间):
```yaml
coarse_eval:
  num_perf_trials: 3
  warmup_min_iters: 2
  warmup_min_time: 0.02
  inner_loop_min_time: 0.002
  profile_custom_model: false
  test_timeout: 40          # 坏 kernel 快速判死,避免 v4 式 ~100 s 拖累
```

**精测配置**:沿用现有 `run.yaml` 默认(`num_perf_trials=100` + profiling),**只对粗筛 top-K** 跑。

### 2.3 实现要点

- **插入点**:[controller.py:651-691](kernelfoundry.internal/kernelfoundry/kernelfoundry/algorithm/controller.py#L651)。当前 `branches_per_iteration` 个候选并行后直接进 `evaluate_batch` 满配评估;改为先 `evaluate_batch(coarse_cfg)` → 排序 → 选 `coarse_top_k` → 再 `evaluate_batch(fine_cfg)`。
- **进化只用精测结果**:写入 MAP-Elites 数据库 / 更新 `best_program` 用**精测**的 `runtime_improvement`(精测才准),粗测仅用于"谁进决赛"。
- **correctness 短路**:粗筛阶段 correctness 失败的候选直接淘汰、不进精测,天然规避 v4 那种"坏候选拖满 timeout 又占 perf 预算"。
- **粗测噪声容忍**:粗测 3 trial 的 std 较大,只用于**粗排**(区分 1× / 5× / 30× 量级足够),不用于精确比较 0.62 vs 0.64 ms。

### 2.4 进一步:把"粗→精"上升到搜索阶段层面(可选)

不止单 trial 内分级,整次运行也可分两段(呼应"先粗后精"):
- **探索段(前 ~60% trial)**:高 `branches_per_iteration`、高 temperature、粗评估 → 广撒网,尽快有 trial 命中高峰 cell;
- **精炼段(后 ~40% trial)**:低 temperature、只从**高峰精英**采样父代(parent 仅取 `get_top_programs` top-K)、满配精测 → 在高峰内压榨(把 v2/v3 的 ~1.0 ms 推向 v5 的 0.62 ms)。

### 2.5 新增配置
```yaml
coarse_to_fine:
  enable: true
  coarse_top_k: 2          # 每 iter 进精测的候选数(branches=4 时筛掉一半)
  coarse_eval: { num_perf_trials: 3, warmup_min_iters: 2, warmup_min_time: 0.02,
                 inner_loop_min_time: 0.002, profile_custom_model: false, test_timeout: 40 }
  explore_fraction: 0.6    # 前 60% trial 为探索段
  refine_parent_top_k: 4   # 精炼段父代只从 top-K 精英采样
```

### 2.6 预期收益
- 单 iter 评估时间约 **-50%~-70%**(只有 top-K 进满配 + profiling);
- 省下的预算回填给**更高的 `branches_per_iteration`**(更多采样 = 更可能命中高峰,见 §4);
- 坏候选(尤其 v4 式)被廉价快速淘汰,不再霸占 perf 预算。

---

## 3. 更有效的 Prompt 提示

### 3.1 现状的三个弱点

1. **tip 太少且随机**:[template_manager.py:183](kernelfoundry.internal/kernelfoundry/kernelfoundry/algorithm/prompts/template_manager.py#L183) 用 `np.random.choice(..., n_tips, replace=False)`,默认 `n_tips=2`,从 OCL 的 ~18 条**通用** tip 里**无种子**随机抽 2 条。→ 每次喂给 LLM 的指导既少又飘(实测 3 个不同 prompt md5)。
2. **无 XMX/DPAS 专门指导**:OCL/SYCL tip 只有泛泛的 vectorize / sub-group / bank-conflict,**没有一条针对 Battlemage XMX/DPAS、SLM 双缓冲、K-tile 流水、`intel_sub_group_block_read`** 的硬件先验。而 v5 之所以破纪录,正是因为它(偶然)抽到了 prefetch + SLM striding + double-buffer 这组 tip。
3. **反馈回路弱**:`use_feedback_llm=False`(默认,[run.yaml:54](kernelfoundry.internal/configs/run.yaml#L54)),且 prompt 里不回传 **profiling 数据**(只有源码 + runtime + console)。LLM 拿不到"瓶颈在哪"的定量信号,只能盲猜。

### 3.2 方案 A:固定 + 扩充高质量 tip(低成本,先做)

- **加 `n_tips`**:从 2 提到 **5–6**(OCL 有 ~18 条,断言 [template_manager.py:62](kernelfoundry.internal/kernelfoundry/kernelfoundry/algorithm/prompts/template_manager.py#L62) `n_tips < min(len)` 仍满足)。信息量↑。
- **可关随机**:加 `tips_deterministic` 开关——为 true 时取**固定的高优先 tip 子集**(或按 trial 轮转),消除"prompt 本身的随机性"这一层(配合 §5 可复现)。
- **新增 B580/XMX 专用 tip 段**(在 [languages.py](kernelfoundry.internal/kernelfoundry/kernelfoundry/algorithm/prompts/languages.py) 的 OCL 列表里补,或单开 `BMG_DPAS_TIPS`),把实测有效的模式固化为先验:
  - 用 `intel_sub_group_block_read_us/_uc` 做 A/B 的向量化 SLM/global 读;
  - SLM 上对 A-tile 双缓冲(double buffer)+ 每 K-iter 一次 barrier 流水;
  - 选对 sub-group 尺寸并用 `intel_reqd_sub_group_size`,让 DPAS 吃满 8×16 片;
  - 三级 prefetch(L1/L2 距离 = prefetch×unrollK,L3 长距离协作 prefetch);
  - 调 SLM stride 避免 bank conflict;寄存器分块累加 C。

### 3.3 方案 B:把 profiling 数据喂回 LLM(中成本,高收益)

- 现在 prompt 的反馈([template_manager.py:199-221](kernelfoundry.internal/kernelfoundry/kernelfoundry/algorithm/prompts/template_manager.py#L199))只含 `code + result + console_output`,**不含 profiler 数据**(EvalResult 里其实有 `profiler_data`)。
- 改进:精测阶段(§2)对**高峰候选**采集的 occupancy / 带宽 / DPAS 利用率等,摘要后注入"上一个 kernel 的瓶颈分析"段。让 LLM 从"盲改"变"看着 roofline 改"。
- 开 `use_feedback_llm=true` 并定制 [feedback_llm_prompt.j2](kernelfoundry.internal/kernelfoundry/kernelfoundry/algorithm/prompts/templates/feedback_llm_prompt.j2),让反馈 LLM 专门输出"下一步该动哪个优化维度"。

### 3.4 方案 C:高峰 kernel 作 few-shot 种子(高收益,治本)

- 把已验证的 **v5 最佳 kernel(0.623 ms / 54.4×)** 存为 seed,入库(`store_generated_kernels_in_db` / `kernels_db_readonly_path`)或作为 prompt 的 "Best Kernel So Far" / RAG 示例([main_prompt.j2](kernelfoundry.internal/kernelfoundry/kernelfoundry/algorithm/prompts/templates/main_prompt.j2) 已有该 section)。
- 效果:每次运行不再从空存档"重抽彩票",而是**从高峰附近开始精炼**——直接把双峰分布的低峰那 ~50% 概率抹掉。这是对"单次结果不可信"最直接的解法。
- 与 §1(C) 的 `reseed` 动作联动:绝望运行直接注入该种子自救。

### 3.5 LLM 采样参数(配合 §5)

[gnai_inference.py:411-425](kernelfoundry.internal/kernelgen/gnai_inference.py#L411) 的 `invoke_model` body **无 `seed` 字段**,且 temperature 仅在显式传入时生效。建议:
- body 增加 `seed`(Bedrock/Anthropic 支持时)→ 可复现;
- 探索段用较高 temperature(多样性),精炼段降到 0.0–0.1(稳定收敛)——按 §2.4 分段设置。

---

## 4. 多次运行 + 预算再分配(把分布讲清楚)

实验铁律:**单次结果不可信**(1×~54× 都可能)。方法层面必须:
- **跑 N 次取分布**(≥3–5),报告 min / median / max + 高峰命中率,而非单个数字;
- **预算再分配**:§1 早停 + §2 粗筛省下的时间,用于
  - 提高 `branches_per_iteration`(每次更可能命中高峰),和/或
  - 增加独立运行次数 N(`1-(1-p)^N`,p≈0.5 时 N=4 命中率已 ~94%);
- **自动 best-of-N**:外层脚本跑 N 次、自动取全局最佳 kernel 落库,对用户只暴露一个稳定的高峰结果。

---

## 5. 可复现性开关(可选,用于调试/对照实验)

为支持"控制变量"的对照实验,提供一键确定化:
- **LLM**:`gnai_inference.py` body 传 `seed` + 固定 `temperature`;
- **Prompt tip**:§3.2 的 `tips_deterministic=true`;
- **框架 RNG**:`random_seed=42` 已 seed numpy/python([evolve_database_optimization_aware.py:741](kernelfoundry.internal/kernelfoundry/kernelfoundry/algorithm/evolve_database_optimization_aware.py#L741)),但要确保 tip 抽样也走这个 seed(目前 `np.random.choice` 受全局 numpy 状态影响,跨 task_name 会漂)。
> 注:生产搜索应保持随机(多样性是命中高峰的来源);确定化仅用于"验证某个改动是否真的有效"的对照实验。

---

## 7. 减少单次迭代耗时 / 加速收敛的其他策略

§1–§2 解决的是"**少跑、跑便宜**"(更少 trial、更便宜的评估)。本节针对另一条正交的轴:**每个 trial 的墙钟时间本身**,以及**用更少 trial 就逼近高峰**。下面的瓶颈均已读码核实。

### 7.1 削减 LLM 推理墙钟(往往是单 trial 最长的一段)

实测 v4 单次 Bedrock 推理高达 **163 s**,是拖慢整次运行的主因之一。三个可直接削减的点:

**(a) 开启 prompt caching(收益最大,改动最小)**
- 现状:[gnai_inference.py:411-425](kernelfoundry.internal/kernelgen/gnai_inference.py#L411) 的 `invoke_model` body **完全没有 `cache_control`**。而 prompt 里**静态不变的大块**(system 指令、硬件 specs、reference 源码、format 模板,见 [main_prompt.j2](kernelfoundry.internal/kernelfoundry/kernelfoundry/algorithm/prompts/templates/main_prompt.j2))**每次调用全量重发、重新预填充**。
- 改进:对 system + reference + 格式说明这些**跨 trial / 跨 branch 不变**的段落打 `cache_control: ephemeral`。Anthropic prompt caching 命中后,这部分的**输入 token 计费降到 ~1/10、TTFT(首 token 延迟)显著下降**。
- 适配性极好:本搜索就是"同一大段上下文 + 每次只换尾部 last_kernel/feedback",是 prompt caching 的理想场景。

**(b) 降 `max_tokens` 上限**
- 现状:[run.yaml:59](kernelfoundry.internal/configs/run.yaml#L59) `max_tokens: 2000`(早先实验里曾用到 10000)。kernel 源码通常 200–400 行,输出 token 越多、解码时间越长。
- 改进:把 `max_tokens` 收到刚好够一个 kernel + 简短分析(如 4000 封顶但配合 stop sequence),避免模型啰嗦长篇分析拖长解码。

**(c) 并行化多 completion(若用 `num_completions>1`)**
- 现状:[gnai_inference.py:426](kernelfoundry.internal/kernelgen/gnai_inference.py#L426) `outputs = [self._invoke_with_retry(body) for _ in range(n)]` 是**串行 for 循环**。
- 改进:同文件已 import `ThreadPoolExecutor`,把这 `n` 个 completion 改成并发提交即可。当前默认 `num_completions=1`([run.yaml:62](kernelfoundry.internal/configs/run.yaml#L62))无影响,但一旦调大就线性变慢——提前修掉。

### 7.2 打破"生成 → 评估 → 下一 trial"的流水线气泡(收益大)

- 现状:[controller.py:652-691](kernelfoundry.internal/kernelfoundry/kernelfoundry/algorithm/controller.py#L652) 是**两道串行屏障**:① 一个 trial 的 `branches_per_iteration` 个候选**全部生成完**(L652-654)→ ② 才进 `evaluate_batch` **全部评估完**(L691)→ ③ 才开始下一 trial 的生成。
- 问题:生成阶段 GPU 空闲(在等 Bedrock),评估阶段 LLM 空闲(在跑 GPU)。两种资源**交替闲置**,单 trial 墙钟 ≈ 生成时间 + 评估时间(相加,而非重叠)。
- 改进:**生成与评估流水线化(pipeline)**——候选 A 一旦生成完就立刻送评估,同时候选 B 还在生成;甚至 trial N 的评估与 trial N+1 的"基于当前最佳的生成"重叠。理想情况单 trial 墙钟 ≈ max(生成, 评估) 而非二者之和,**省下约等于较短那一段的时间**。
- 实现:把 L652 的"生成线程池"与 L868 的"评估线程池"用一个生产者-消费者队列串起来(生成 future 完成即 enqueue 评估),而不是两个 `as_completed` 屏障。

### 7.3 候选去重 / 评估记忆化(省冗余 GPU 时间)

- 现状:[controller.py](kernelfoundry.internal/kernelfoundry/kernelfoundry/algorithm/controller.py) 里**没有任何 kernel 源码 hash / dedup / 评估缓存**(grep `md5|hash|seen|cache|dedup` 在 controller.py 无命中)。
- 问题:低 temperature(精炼段 temp→0)时,LLM 很容易**生成与上一版几乎相同甚至完全相同**的 kernel,却仍走一遍完整 build + correctness + 100 次 perf 测量。
- 改进:对 normalize 后(去注释/空白)的 kernel 源码算 hash,维护一张 `hash → EvalResult` 表;命中则**直接复用结果、跳过整个评估**。配合 §7.2 的低 temp 精炼段尤其有效。

### 7.4 缓存 reference 的构建与测量(省每次启动的固定开销)

- 现状:reference 的测量结果在**一次运行内**已缓存([test_custom_task.py:87](kernelfoundry.internal/kernelfoundry/kernelfoundry/eval_pipeline/tasks/test_custom_task.py#L87) `task.test_result_reference is None` 才重测),但**每次新运行启动都重新 build + correctness + benchmark 一遍 reference**(实测每次启动那段 ~33.9 ms 基线测量 + 两轮 correctness)。
- 改进:reference 的 build artifact 与 `ref_speed=33.9 ms` 对(固定硬件 + 固定 shape)**跨运行持久化缓存**(按 `gpu_arch + shape + ref_src_hash` 做 key)。§4 的 best-of-N 要跑 N 次,这段固定开销能省 N-1 次。

### 7.5 编译/构建提速(省每个候选的 build 段)

- `build_timeout: 200`([run.yaml:78](kernelfoundry.internal/configs/run.yaml#L78))是上限;OCL 在线编译本身不慢(实测 build 'custom' ~0.27 s),但坏 kernel 可能触发长编译。可:
  - 对 OCL kernel 开启 **二进制缓存**(NEO 的 `cl_cache` / 离线 `ocloc` 编译产物缓存),相同源码不重复编译;
  - 缩短 `build_timeout` 让病态编译快速失败(配合 §2.2 粗筛短 timeout)。

### 7.6 用更少 trial 逼近高峰(加速"收敛"本身)

除了"每 trial 更快",还能"更少 trial 到高峰":

- **岛屿/小批量早期淘汰**:一个 trial 内并行 `branches_per_iteration` 个候选,用 §2 的粗测**立刻砍掉明显差的**,把精测预算集中给有希望的——等效于在相同墙钟里探索更多方向。
- **温度退火(annealing)**:探索段高 temp 广撒网快速命中高峰 cell,精炼段 temp→0 快速收敛(§2.4 / §3.5 已提)。退火比全程固定 temp 收敛更快。
- **高峰种子 + warm start**(§3.4):直接从已验证高峰 kernel 起步,跳过"从低峰爬出来"的多个 trial——这是把"收敛 trial 数"从 ~6 压到 1–2 的最直接手段。

> 小结:§7.1(prompt caching)+ §7.2(流水线)是**单 trial 墙钟**的两个最大杠杆,且都不改变搜索语义、零风险;§7.3/7.4/7.5 是省冗余;§7.6 是减少所需 trial 数。

---

## 6. 落地优先级与预期

| 优先级 | 改动 | 成本 | 预期收益 |
|---|---|---|---|
| **P0** | §1 早停(A 平台 + C 绝望) | 低(controller.py 加 ~30 行 + config) | 省 30–40% trial;v4 式运行止损 |
| **P0** | §3.2 加 tip 数 + B580/XMX 专用 tip | 低(改 languages.py / config) | 直接抬高单次命中高峰概率 |
| **P0** | §7.1(a) 开 prompt caching | 低(gnai_inference.py 加 cache_control) | LLM 输入 token ~1/10、TTFT↓,**直削单 trial 最长段** |
| **P1** | §2 粗→精分级评估 | 中(controller.py 评估流程改造) | 单 iter -50~70% 评估时间 |
| **P1** | §7.2 生成/评估流水线化 | 中(队列串联两线程池) | 单 trial 墙钟 ≈ max(生成,评估) 而非相加 |
| **P1** | §3.4 高峰 kernel 作种子(= §7.6 warm start) | 中(入库 + prompt 注入) | 抹掉低峰 ~50% 概率 + 收敛 trial 数 6→1~2 |
| **P2** | §7.3 候选去重 / 评估记忆化 | 低中(hash 表) | 省低 temp 段的冗余评估 |
| **P2** | §7.4 reference 跨运行缓存 | 低中 | best-of-N 省 N-1 次基线开销 |
| **P2** | §3.3 profiling 喂回 + feedback_llm | 中高 | 从盲改到定向优化 |
| **P2** | §4 best-of-N 外层 + §5 可复现开关 | 低 | 稳定交付 + 可做对照实验 |

**综合目标**:用 §1+§2 把单次运行的**预算砍掉一半**,§7 再把**每个 trial 的墙钟**压下来(prompt caching + 流水线 + warm start),把省下的预算按 §4 投到**更多采样 / 更多次运行**;用 §3 把**单次命中高峰的基础概率**抬上去。多者叠加,目标是把"同命令命中高峰"从实测 ~50% 提升到稳定 ≥90%、把"收敛所需 trial 数"从 ~6 压到 1–2,并让高峰内性能从 v2/v3 的 ~21 TFLOPS 稳定逼近 v5 的 ~34.5 TFLOPS 乃至更高。

---

## 附:关键代码锚点(改动落点)

- 主搜索循环 / 早停插入点:[controller.py:615](kernelfoundry.internal/kernelfoundry/kernelfoundry/algorithm/controller.py#L615)(loop)、[:798](kernelfoundry.internal/kernelfoundry/kernelfoundry/algorithm/controller.py#L798)(唯一既有 break)
- 候选生成 + 评估(粗→精插入点):[controller.py:651-691](kernelfoundry.internal/kernelfoundry/kernelfoundry/algorithm/controller.py#L651)
- 评估成本配置:[run.yaml:96-102](kernelfoundry.internal/configs/run.yaml#L96)(`num_perf_trials/warmup_*/inner_loop_min_time/profile_*`)、`test_timeout`/`build_timeout`([:79](kernelfoundry.internal/configs/run.yaml#L79)/[:78](kernelfoundry.internal/configs/run.yaml#L78))
- 自适应 warmup/trials 计算:[performance.py:251-276](kernelfoundry.internal/kernelfoundry/kernelfoundry/eval_pipeline/utils/performance.py#L251)
- prompt 组装:[template_manager.py:75-176](kernelfoundry.internal/kernelfoundry/kernelfoundry/algorithm/prompts/template_manager.py#L75)、模板 [main_prompt.j2](kernelfoundry.internal/kernelfoundry/kernelfoundry/algorithm/prompts/templates/main_prompt.j2)
- tip 定义 / 随机抽样:[languages.py:65](kernelfoundry.internal/kernelfoundry/kernelfoundry/algorithm/prompts/languages.py#L65)(OCL ~18 条)、[template_manager.py:183](kernelfoundry.internal/kernelfoundry/kernelfoundry/algorithm/prompts/template_manager.py#L183)(`n_tips` 默认 2)
- 反馈回路:[template_manager.py:199-221](kernelfoundry.internal/kernelfoundry/kernelfoundry/algorithm/prompts/template_manager.py#L199)、`use_feedback_llm` [run.yaml:54](kernelfoundry.internal/configs/run.yaml#L54)
- LLM 调用(无 seed):[gnai_inference.py:411-425](kernelfoundry.internal/kernelgen/gnai_inference.py#L411)
- 框架 RNG seed:[evolve_database_optimization_aware.py:741](kernelfoundry.internal/kernelfoundry/kernelfoundry/algorithm/evolve_database_optimization_aware.py#L741)
