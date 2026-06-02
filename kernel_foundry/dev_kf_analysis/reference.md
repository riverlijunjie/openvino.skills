# KernelFoundry-Agent 设计文档

> 把 KernelFoundry 从「弱模型 + 盲进化的 batch 实验框架」重构为「强模型(Claude)+ 工具 + 领域 Skill 的 E2E 闭环优化 sub-agent」。
>
> 版本 v0.1 · 2026-05-29 · 状态:草案,待评审

---

## 1. 背景与动机

现有 `kernelfoundry.internal` 在实测中暴露 4 个问题(本文档的设计目标即逐一消除):

| # | 现有问题 | 根因 | 本设计的解法 |
|---|---|---|---|
| 1 | Task 配置极复杂(Hydra 多层 yaml) | 为 batch 论文实验设计,非用户需求驱动 | NL intent → task spec,agent 自动解析 |
| 2 | Token 消耗大、kernel 质量反而差于 CC | 弱 base model(GNAI/Denvr)+ 盲采样进化;serial-refine > parallel-sample 已被 Kevin-32B 证实 | 强推理模型 + 带反馈的迭代代替盲采样 |
| 3 | 无法接 E2E loop | KernelBench 范式 = 孤立算子 @ 固定 shape,与真实 workload/runtime 脱节 | orchestrator 编排 profile→generate→integrate→verify→iterate 闭环 |
| 4 | MAP-Elites 锁死优化方向 | behavior descriptor 手工固定;单最优目标下 QD 多样性开销是浪费 | MAP-Elites 降级为可选 Skill,descriptor 自适应 |

**核心理念转变:** 从 *"训练/驱动一个 kernel 模型"* → *"用 harness 把强通用模型武装成 kernel 专家"*。护城河从「模型」转移到「Intel HW 领域 Skill + 可验证的工具闭环」。

---

## 2. 目标 / 非目标

### 2.1 目标(Goals)
- **G1** 用户用自然语言描述需求即可启动(「把这个 OV 模型里最慢的算子在 BMG 上优化一下」)。
- **G2** 端到端闭环:分析真实 inference workload → 定位 kernel 热点 → 生成 → 集成进 runtime → 量 E2E perf → 迭代。
- **G3** 同 request 下,相比现有 KF **token 更省、kernel 性能更好**(可量化对比)。
- **G4** 搜索策略可插拔;MAP-Elites 仅在需要时由 agent 主动调用。
- **G5** 验证不可被 reward-hack(吸取 Sakana 教训)。
- **G6** 复用 KF 已有硬资产(build/test harness、unitrace、RAG 语料、program DB),不重造轮子。

### 2.2 非目标(Non-Goals)
- 不训练专用 kernel 模型(那是 Kevin/KernelLLM 路线;本设计走 agent 路线)。
- 不追求一次覆盖所有 runtime;首期只打通 **1 个** backend(见 §13 决策点)。
- 不替换 KernelBench/robust-kbench;它们保留为回归 harness(§11)。
- 不做通用代码生成;专注 GPU kernel(SYCL 优先,CUDA/Triton 兼容)。

---

## 3. 设计原则

1. **Grounded over generative** — 任何 perf/correctness 结论必须来自确定性 Tool 的真实测量,模型不得自行宣称。
2. **Reasoning over sampling** — 默认带反馈的串行 refine;广撒网采样仅作为可选策略。
3. **Expertise as Skills** — Intel HW 知识、SYCL idiom、调优经验做成可组合、可缓存的 Skill。
4. **Verification first** — 验证是核心 Tool,不是事后步骤;先有防作弊的 verify,才允许任何 perf 优化。
5. **E2E truth** — 最终指标是真实 workload 的端到端性能,micro-benchmark 只是中间信号。
6. **Reuse, don't rebuild** — KF 的 build/profile/RAG/DB 直接当 Tool 库调用。

---

## 4. 架构总览

```
                          ┌─────────────────────────────────────────┐
   用户 NL 需求  ───────▶  │      Orchestrator (Claude sub-agent)      │
   "优化这个 OV 模型在      │   意图解析 · 计划 · 闭环编排 · 迭代决策      │
    BMG 上的瓶颈算子"       └───────────────┬───────────────────────────┘
                                          │ 调用
        ┌─────────────────────────────────┼──────────────────────────────────┐
        ▼                                 ▼                                  ▼
┌───────────────┐              ┌───────────────────┐              ┌──────────────────┐
│    TOOLS      │              │      SKILLS       │              │     MEMORY       │
│ (确定性/可验证) │              │ (领域专长/可缓存)   │              │  (跨 run 学习)    │
├───────────────┤              ├───────────────────┤              ├──────────────────┤
│ profile_workload            │ intel-hw-spec     │              │ kernel_memory    │
│ build_kernel  │              │ sycl-optimization │              │ (program DB:     │
│ verify_correctness          │ perf-tuning-guide │              │  kernel+E2E perf │
│ benchmark_e2e │              │ search-strategies │              │  +workload ctx)  │
│ integrate_runtime           │ runtime-integration│             │  RAG 检索        │
│ collect_perf_metrics        │ (MAP-Elites=可选)  │              └──────────────────┘
└───────┬───────┘              └───────────────────┘
        │ 复用 KF
        ▼
┌─────────────────────────────────────────────────────────────┐
│  KF 硬资产:TorchCompiler · docker(sycl-all/ptl)· unitrace   │
│  eval_helper · perf_score schema · RAG 语料 · program DB tables│
└─────────────────────────────────────────────────────────────┘
```

三层职责:
- **Tools** = 模型的"手和眼"——确定性、可验证,返回真实测量值。
- **Skills** = 模型的"专业知识"——静态、可 prompt-cache,提供 Intel HW 调优专长。
- **Memory** = 模型的"经验"——跨 run 累积 kernel + 真实 perf + workload 上下文,支持 warm-start。

---

## 5. E2E 控制循环

Orchestrator 的主状态机:

```
[1] INTENT      解析 NL 需求 → {workload, target HW, 目标(latency/throughput), 约束}
      │
[2] PROFILE     profile_workload(workload, HW) → 逐算子耗时 + roofline 归因
      │         collect_perf_metrics → 找 top-N 热点 & bound 类型(mem/compute)
      ▼
[3] SELECT      按「耗时占比 × 可优化空间 × 可替换性」选目标算子
      │         查 kernel_memory:是否已有该算子在该 HW 的优化历史?→ warm-start
      ▼
[4] STRATEGIZE  查 perf-tuning-guide + intel-hw-spec → 决定优化方向
      │         选搜索策略(默认 greedy-refine;空间宽时才上 MAP-Elites)
      ▼
[5] GENERATE    Claude 生成/修改 kernel(注入 sycl-optimization + HW spec + RAG 范例)
      │
[6] BUILD       build_kernel(容器化编译) ── 失败 ─▶ 反馈错误回 [5]
      │ 成功
[7] VERIFY      verify_correctness(多 shape + 随机输入 + 防作弊) ── 不过 ─▶ 回 [5]
      │ 通过                                              (防 reward-hacking,见 §9)
[8] MICRO-BENCH  micro 层面测 kernel runtime(快速信号,非最终指标)
      │
[9] INTEGRATE   integrate_runtime(注册 custom op / 替换 primitive / 热加载)
      │
[10] E2E-BENCH  benchmark_e2e(真实 workload 端到端) ── 退化 ─▶ 回退 & 反馈回 [4/5]
      │ 提升
[11] RECORD      kernel_memory 存档(kernel + E2E perf + 上下文)
      │
[12] DECIDE      达标? ─是─▶ 返回最优 + 报告
                 否 ─▶ 回 [3](下一个热点)或 [4](同算子换方向),受 budget 约束
```

关键差异 vs KF:**[2][9][10] 是 KF 完全没有的层**(真实 workload profiling、runtime 集成、E2E 验证);**[5] 用推理代替 KF 的盲采样**;**[4] 的搜索策略是可选的,MAP-Elites 不再是强制主循环**。

---

## 6. Tools 接口定义

> 约定:所有 Tool 确定性、可独立测试;perf 类返回带统计(mean/p50/p95、trials);失败返回结构化错误供模型反馈。

### 6.1 `profile_workload`
```
in:  { workload_ref, hw_target, runtime: "openvino"|"pytorch-xpu"|"onednn", input_spec }
out: { per_op: [{op_name, op_type, shape, time_us, pct_total, bound: "mem"|"compute"|"latency"}],
       total_us, roofline: {achieved_gflops, achieved_bw, peak_gflops, peak_bw} }
复用: unitrace + (OV perf counters / oneDNN verbose / torch profiler)
```

### 6.2 `collect_perf_metrics`
```
in:  { kernel_ref | op_ref, hw_target }
out: { occupancy, mem_bw_util, compute_util, cache_hit, stall_reasons[], roofline_point }
复用: unitrace 原始 profiler_data
```

### 6.3 `build_kernel`
```
in:  { kernel_src, language: "sycl"|"cuda"|"triton"|"ocl", hw_arch, build_flags? }
out: { ok: bool, artifact_ref?, compile_log, errors: [{file,line,msg}] }
复用: KF TorchCompiler + docker 镜像(kernelfoundry_sycl-all / -ptl)
约束: build_timeout 复用 KF 配置
```

### 6.4 `verify_correctness`  ★ 核心,见 §9
```
in:  { kernel_ref, ref_impl(pytorch), shapes: [多个], dtypes, n_random_trials }
out: { perf_score: 0..5, correct: bool,
       checks: {shape_match, value_match(rtol/atol), anti_gaming_passed},
       failures: [{shape, max_abs_err, kind}] }
复用: KF eval_helper + perf_score schema,但强化多 shape & 防作弊
```

### 6.5 `benchmark_e2e`
```
in:  { workload_ref, hw_target, runtime, n_warmup, n_trials }
out: { e2e_latency_us: {mean,p50,p95}, throughput, vs_baseline_speedup,
       per_op_delta: [...]  // 集成新 kernel 后逐算子变化
     }
新增: 这是 KF 没有的层 —— 真实 workload 端到端
```

### 6.6 `integrate_runtime`  ★ 最难,backend-specific
```
in:  { kernel_artifact, op_target, runtime, integration_mode }
out: { ok, integrated_runtime_ref, rollback_token, modified_files? }
模式: openvino → custom op / extension
      pytorch-xpu → torch custom op 注册 (TORCH_LIBRARY)
      onednn → custom primitive / brgemm 替换
注意: "有时需改 runtime 加载新算子" 即在此层;必须支持 rollback
```

### 6.7 `kernel_memory`
```
put: { op_signature, hw, kernel_src, micro_perf, e2e_perf, workload_ctx, strategy_used }
get: { op_signature, hw } → [历史最优 kernel + perf]  // warm-start / RAG
复用: KF program DB tables + RAG 检索
```

---

## 7. Skills 目录

> Skill = 静态领域知识 + 决策指引,可 prompt-cache。这是 Intel 的真正护城河。

| Skill | 内容 | 来源 |
|---|---|---|
| **intel-hw-spec** | Arc/BMG/PTL 的 Xe 架构、EU/XVE 数、SIMD 宽度(FP32=8/FP16=16)、内存层级与带宽、SLM 大小、sub-group size、硬限制 | 新建,部分自 KF `prompts/hardware/` |
| **sycl-optimization** | sub-group ops(`reduce_over_group`/`group_broadcast`,禁用 `.shuffle()`)、local memory tiling、coalesced access、bank conflict、vectorization、idiom 与反模式 | 扩展 KF `languages.py` 的 SYCL tips |
| **perf-tuning-guide** | roofline 驱动决策树:mem-bound→提升复用/coalescing/SLM tiling;compute-bound→ILP/unroll/vectorize/sub-group;latency-bound→fusion/减同步 | 新建 |
| **search-strategies** | greedy-refine(默认,最省)/ beam / parallel-sample / **MAP-Elites(可选,descriptor 自适应)**;agent 按问题选 | MAP-Elites 自 KF `qd_gradient.py` + `map_elites_patterns.py`,但改为可调用 |
| **runtime-integration** | 每后端一份 how-to:OV extension 注册流程 / torch custom op / oneDNN primitive 替换;含「何时需要改 runtime」 | 新建(随 §13 决策的首个后端先写一份) |

**MAP-Elites 的新定位:** 不再是主循环,而是 `search-strategies` 里的一个选项。仅当 agent 判断「优化空间宽、需要 stepping stones 跳出局部最优」时调用;且 behavior descriptor 由 agent 按当前算子动态定义,而非硬编码。这直接修掉问题 #4。

---

## 8. Memory / Program DB

- 复用 KF 的 program DB schema(`database/tables.py` 的 `Kernel` 表),扩展字段:`e2e_perf`、`workload_ctx`、`strategy_used`、`hw_arch`。
- **Warm-start**:新任务先查 `kernel_memory(op_signature, hw)`,命中则以历史最优为起点,而非从零生成 → 进一步省 token。
- **RAG**:保留 KF 的 HecBench/ESIMD/joint_matrix/pytorch→sycl 语料(SYCL 稀缺语料,很值钱),作为 GENERATE 阶段的 few-shot 来源。
- 跨 run 学习:记录「哪类 strategy 对哪类 bound 的算子有效」,反哺 `perf-tuning-guide`。

---

## 9. 验证设计(防 Reward-Hacking)★

> Sakana 翻车的根因:eval 被 reward-hack(memory reuse、返回常量、跳过计算)。agent 盯着 perf 数字优化时**一定会**学会作弊。此节是整个系统可信度的基石。

`verify_correctness` 必须满足:

1. **多 shape + 多 dtype**:不止预定义 shape;覆盖边界(非 2 的幂、非对齐、含 batch=1)。
2. **随机化输入**:每次重新随机生成输入张量,禁止依赖固定常量 → 防「返回期望常量」。
3. **防内存复用**:每次运行前重置/污染输出 buffer;隔离上一次 PyTorch run 的结果 → 防「复用上一轮结果」。
4. **强制计算证据**:对比 NCU/unitrace 的实际算力/访存计数,若「号称算了但 profiler 显示几乎没算」→ 判作弊(直击 Sakana 的 `Conv3d` 跳过卷积案例)。
5. **E2E 输出等价**:集成后,真实 workload 的最终输出需与 baseline 数值等价(rtol/atol),不只算子级。
6. **沙箱隔离**:致命错误(illegal memory access)不得污染后续测量(借 Kevin 的 sandbox 思路)。

验证不过 → 直接打回 GENERATE,**绝不让未验证的 kernel 进入 perf 排序**。

---

## 10. Token / 成本策略

- **Prompt caching**:`intel-hw-spec` / `sycl-optimization` / system prompt 等大块静态内容固定缓存,跨迭代复用。
- **分层模型**:机械子任务(profiler 输出解析、代码抽取、日志归纳)用便宜模型;推理与 codegen 用 Claude。
- **Warm-start**:命中 memory 则改写历史最优,避免从零生成。
- **串行优先**:默认 greedy-refine(少量高质量迭代),避免 KF 的盲采样 token 黑洞。
- **Budget 约束**:每个任务设 token/迭代上限,DECIDE 阶段据此决定继续/收敛。

---

## 11. 复用 vs 替换(迁移清单)

| KF 组件 | 处置 | 说明 |
|---|---|---|
| `TorchCompiler` + docker 镜像 | ✅ 复用 → `build_kernel` | 直接当 Tool |
| unitrace 集成 | ✅ 复用 → `profile_workload` / `collect_perf_metrics` | |
| `eval_helper` + `perf_score` schema | ✅ 复用 + 强化 → `verify_correctness` | 加多 shape & 防作弊 |
| RAG 语料(HecBench/ESIMD/...) | ✅ 复用 → Memory/RAG | SYCL 稀缺语料,资产 |
| program DB tables | ✅ 复用 + 扩字段 → `kernel_memory` | 加 e2e_perf/workload_ctx |
| `languages.py` HW tips | ✅ 提炼 → `sycl-optimization` skill | |
| `qd_gradient` / `map_elites_patterns` | ⚠️ 降级 → `search-strategies` 里的可选项 | descriptor 改自适应 |
| Hydra `configs/*` task 配置 | ❌ 替换 → NL intent | |
| GNAI/Denvr 盲采样主循环 | ❌ 替换 → Claude 推理闭环 | |
| RabbitMQ/Celery 分布式队列 | ◽ 可选保留 | 仅大规模批量时需要;PoC 不需要 |
| KernelBench/robust-kbench | ✅ 保留为回归 harness | 见 §12 |

---

## 12. 评测方法

- **主指标**:真实 workload 的 **E2E 端到端 perf**(latency p50/p95、throughput),vs PyTorch eager / torch.compile / OV 默认。
- **回归 harness**:robust-kbench 跑算子级正确性 & 防 gaming(防退化)。
- **对照实验**(验证 G3):同一 request,分别用 (a) 现有 KF、(b) KernelFoundry-Agent,对比 **token 消耗** 与 **最终 kernel 性能**。这是证明重构价值的关键数据。
- **消融**:有/无 `intel-hw-spec` skill、有/无 warm-start、greedy vs MAP-Elites,量化各组件贡献。

---

## 13. 分阶段路线图

| 阶段 | 内容 | 产出 / 验收 |
|---|---|---|
| **P0 决策** | 选定首个 runtime(见下方决策点);定 1 个目标 workload + 1 个目标 HW | 决策记录 |
| **P1 Tools 骨架** | `build_kernel`/`verify_correctness`(含防作弊)/`benchmark_e2e` 三件套,复用 KF | 能对单个手写 kernel 完成 build→verify→bench |
| **P2 最小闭环 PoC** | orchestrator 串 `profile→generate(Claude)→build→verify→micro-bench`(暂不集成 runtime) | 对 1 个算子,验证「比 KF 省 token、micro perf 更好」(G3 初证) |
| **P3 Skills** | `intel-hw-spec` + `sycl-optimization` + `perf-tuning-guide` | 消融显示 skill 带来可量化提升 |
| **P4 E2E 集成** | `integrate_runtime`(首个后端)+ `benchmark_e2e` 接入闭环 [9][10] | 真实 workload 上完成完整闭环,E2E 有提升(G2) |
| **P5 Memory & 策略** | `kernel_memory` warm-start + `search-strategies`(含可选 MAP-Elites) | 跨 run 学习生效;MAP-Elites 可按需触发(G4) |
| **P6 NL 前端 & 报告** | intent 解析 + 优化报告 | 用户一句话启动(G1) |

---

## 14. 开放决策点

> 以下需你拍板,**P0 决策点**最关键:

1. **首个 runtime(决定 PoC 形态)** — 推荐排序:
   - **OpenVINO**(推荐):贴合你 OV/PET 主线,集成价值最高;但 custom op/extension 集成较重。
   - **PyTorch-XPU**:`TORCH_LIBRARY` 注册 custom op 最轻,PoC 最快出结果;适合先验证闭环可行性。
   - **oneDNN**:primitive 替换最底层、最通用,但集成最复杂。
   - *建议:P2 PoC 用 PyTorch-XPU 快速跑通,P4 E2E 集成切到 OpenVINO 对齐你的主线。*
2. **首个目标 HW**:BMG / Arc A770 / PTL?(决定 `intel-hw-spec` 先写哪个)
3. **首个目标 workload**:用哪个真实 inference 模型做 E2E?(LLM decode? 某个 OV demo?)
4. **分布式队列**:PoC 阶段是否完全砍掉 RabbitMQ/Celery,只做单机闭环?(建议:砍掉,P5 后视需要再加)

---

## 15. 风险

| 风险 | 等级 | 缓解 |
|---|---|---|
| 验证被 reward-hack,perf 数字造假 | 🔴 高 | §9 全套防作弊;verify 不过绝不进排序 |
| runtime 集成 backend-specific、工作量大 | 🔴 高 | 首期只啃 1 个后端;`integrate_runtime` 支持 rollback |
| SYCL 训练/RAG 语料稀缺 | 🟡 中 | 复用 KF 语料 + 强 HW skill 补偿;memory 跨 run 积累 |
| Claude 生成的 SYCL 在 Intel GPU 上有兼容坑 | 🟡 中 | `sycl-optimization` skill 显式列反模式;build/verify 快速反馈 |
| E2E 提升被其他算子/调度掩盖 | 🟡 中 | `benchmark_e2e` 返回 per_op_delta,归因到具体算子 |

---

## 附:与社区方案的定位

| | 模型 | 搜索 | E2E | HW |
|---|---|---|---|---|
| Kevin-32B | 自训 32B + RL | 多轮 refine | ❌ | NV |
| KernelLLM | 自训 8B SFT | 单 shot | ❌ | NV/Triton |
| Sakana | 现成 LLM | 进化 | ❌ | NV |
| **KernelFoundry-Agent** | **现成强模型 + Skill** | **推理为主,QD 可选** | **✅ 闭环** | **Intel 优先** |

差异化:**唯一做 Intel 多后端 + 真实 E2E 闭环 + agent 范式**的方案。护城河 = Intel HW Skill + 防作弊的可验证工具闭环。


