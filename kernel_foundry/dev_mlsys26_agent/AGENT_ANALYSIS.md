# MLSys26 Agent Framework 分析总结

> 日期：2026-06-03  
> 分析范围：  
> - `/mnt/river/kernel_foundry/mlsys26/mlsys26-agent-baseline`  
> - `/mnt/river/kernel_foundry/mlsys26/flashinfer-bench-starter-kit`  
> - `/mnt/river/kernel_foundry/mlsys26/datasets/mlsys26-contest`  
> 目标：分析 MLSys26 FlashInfer AI Kernel Generation Contest 相关 agent baseline、benchmark starter kit、contest dataset 的架构、组件交互、瓶颈、缺口和改进方向。

## 1. 总体结论

MLSys26 目录下的代码形成了一个较完整但仍偏 baseline 的 AI kernel generation 框架：

1. **`mlsys26-agent-baseline`**：LLM 驱动的 kernel 生成与搜索 agent，主要生成 Triton kernel，通过 FlashInfer-Bench 评估正确性和性能。
2. **`flashinfer-bench-starter-kit`**：比赛官方 starter kit，用于编写 Triton/CUDA solution、打包 `solution.json`、本地或 Modal/B200 上运行 benchmark。
3. **`datasets/mlsys26-contest`**：比赛数据集，采用 FlashInfer Trace 格式，包含 definition、workload 和 FlashInfer baseline solution。

当前 agent baseline 的核心价值是：已经打通了 **自然语言/LLM 生成 kernel → FlashInfer-Bench 确定性评估 → 反馈给 LLM 继续优化** 的闭环。

但从竞赛级别高性能 kernel optimization agent 的角度看，它仍然比较浅：

- kernel generator 以通用 Triton prompt 为主，缺少 MoE/GDN/DSA 任务专用策略；
- search algorithm 主要是 iterative refine 和 evolve prompt-pool，没有真正的多分支/Pareto/descriptor 搜索；
- performance model 只有实测 speedup，没有 surrogate model、profile model 或 cost model；
- testing framework 直接跑 FlashInfer-Bench，没有分层验证、smoke test、sanitizer、NCU profile 闭环；
- artifact 和 memory 记录较基础，缺少 candidate lineage、branch family、per-workload 分析和 official score tracker；
- CUDA/CuTe/DeepGEMM 路径尚未成为一等公民，而 B200/Blackwell 上很多任务可能需要更底层控制。

因此，推荐的演进方向是：将现有 baseline 从“prompt-and-measure loop”升级为一个 **task-aware、profile-guided、multi-branch、official-score-aware 的 kernel optimization agent**。

## 2. 代码库结构

### 2.1 `mlsys26-agent-baseline`

路径：`/mnt/river/kernel_foundry/mlsys26/mlsys26-agent-baseline`

这是主要 agent 框架。

| 文件 | 作用 |
|---|---|
| `agent/main.py` | 主入口；加载配置、加载任务、创建 inference server、创建 eval backend、遍历 task 并保存最终结果。 |
| `agent/iterative_agent.py` | Iterative agent：先 propose 一个 kernel，再用 `str_replace` 进行多轮局部调优。 |
| `agent/evolve_agent.py` | Evolve agent：维护 recent pool 和 elite pool，从上下文中采样已有 kernel，再让 LLM 生成新 proposal。 |
| `agent/eval.py` | 本地评估封装：构造 FlashInfer-Bench `Solution`，调用 `Benchmark`，返回 `EvalResult`。 |
| `agent/modal_eval.py` | Modal/B200 远程评估封装，逻辑基本镜像 `eval.py`。 |
| `agent/api.py` | OpenAI/Anthropic API 客户端创建和重试逻辑。 |
| `agent/utils.py` | 数据集路径、任务加载、代码块提取、edit 提取、字符串替换等工具函数。 |
| `prompt/proposer_prompt.py` | 初始 kernel 生成 prompt。 |
| `prompt/tuner_prompt.py` | 基于 `old_str` / `new_str` 的调优 prompt。 |
| `config/config_iterative.yaml` | iterative agent 默认配置。 |
| `config/config_evolve.yaml` | evolve agent 默认配置。 |
| `config/tasks_default.txt` | 默认任务：`dsa_paged`、`gdn`、`moe`。 |
| `config/tasks_mini.txt` | smoke test：单个 MoE definition。 |

### 2.2 `flashinfer-bench-starter-kit`

路径：`/mnt/river/kernel_foundry/mlsys26/flashinfer-bench-starter-kit`

这是比赛 starter kit，用于实现和提交 solution。

| 文件 | 作用 |
|---|---|
| `README.md` | 比赛说明、环境安装、solution 配置、benchmark 运行方式。 |
| `EVALUATION.md` | 官方评测环境、命令、计分规则。 |
| `config.toml` | solution 名称、definition、author、language、entry point。 |
| `scripts/pack_solution.py` | 将 `solution/triton` 或 `solution/cuda` 打包成 `solution.json`。 |
| `scripts/run_local.py` | 本地打包并运行 FlashInfer-Bench。 |
| `scripts/run_modal.py` | 在 Modal B200 上运行 benchmark。 |
| `solution/triton/kernel.py` | Triton kernel 模板。 |
| `solution/cuda/kernel.cu` | CUDA kernel 模板。 |
| `solution/cuda/binding.py` | CUDA kernel 的 TVM FFI binding 模板。 |

Starter kit 支持：

- Triton solution；
- CUDA solution；
- destination passing style；
- local benchmark；
- Modal B200 benchmark；
- sanitizer 和 NCU profiling API。

### 2.3 `datasets/mlsys26-contest`

路径：`/mnt/river/kernel_foundry/mlsys26/datasets/mlsys26-contest`

该数据集采用 FlashInfer Trace 格式。

统计结果：

| 类型 | 数量 |
|---|---:|
| Kernel definitions | 5 |
| Workload JSONL files | 5 |
| Workload records | 324 |
| Baseline solution JSON files | 5 |

Definition 列表：

| op type | definition | workload 数量 | 说明 |
|---|---|---:|---|
| `moe` | `moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048` | 19 | FP8 block-scale MoE，包含 routing 和两个 grouped GEMM。 |
| `gdn` | `gdn_decode_qk4_v8_d128_k_last` | 54 | Gated Delta Net decode，单 token recurrent state update。 |
| `gdn` | `gdn_prefill_qk4_v8_d128_k_last` | 100 | Gated Delta Net prefill，k-last state layout。 |
| `dsa_paged` | `dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64` | 23 | DeepSeek 风格 paged sparse attention。 |
| `dsa_paged` | `dsa_topk_indexer_fp8_h64_d128_topk2048_ps64` | 128 | DSA TopK indexer，FP8 quantized score computation。 |

## 3. 当前 Agent 架构

### 3.1 主流程

`agent/main.py` 控制整体流程：

```text
读取 config
  -> 读取 tasks list
  -> 创建 LLM inference server
  -> 创建 eval_fn：local 或 modal
  -> 遍历每个 task
       -> 加载 definition JSON
       -> 构造 task_params
       -> 调用 iterative 或 evolve agent
       -> 保存 global_best_kernel_N.py 和 global_best_metrics_N.json
```

关键函数：

- `run_main_loop(args)`：遍历所有 task，统计 correct count、sum speedup、average speedup。
- `run_agent(args, inference_server, level, problem_id)`：对单个 definition 运行 agent。
- `_check_cached_result(args, level, problem_id)`：用于 resume 或跳过已有完整结果。

当前输出结构大致为：

```text
outputs/<agent_type>_<test_source>_<steps>_<timestamp>/
  config.yaml
  <op_type>_<problem_id>/
    reference_src.py
    proposal_*.py
    tune_*.py
    *_metrics.json
    *_prompt.txt
    global_best_kernel_N.py
    global_best_metrics_N.json
```

### 3.2 Kernel Generator

Kernel generator 主要由两个 prompt 组成。

#### Proposer

实现位置：

- `agent/iterative_agent.py::propose_step`
- `prompt/proposer_prompt.py`

输入：

- definition JSON；
- target GPU / architecture；
- dtype 信息；
- recent / elite kernel pool；
- 对应 metrics。

输出：

- 完整 Python/Triton code；
- 默认要求暴露 `run` entry point；
- 通过 `extract_first_code` 提取第一段代码。

优点：

- 简单直接；
- 可以生成完整 kernel；
- 可以参考已有 kernel pool。

问题：

- prompt 比较通用，缺少 MoE/GDN/DSA 专用优化策略；
- 默认偏 Triton，没有把 CUDA/CuTe/DeepGEMM 作为一等搜索路径；
- 对 exact function signature、DPS/value-returning style、workload shape distribution 的约束不够结构化；
- 让 LLM 自己处理 device movement 可能和 FlashInfer-Bench 的评测方式产生额外开销或语义不一致。

#### Tuner

实现位置：

- `agent/iterative_agent.py::refine_step`
- `prompt/tuner_prompt.py`

Tuner 要求 LLM 输出：

```text
<reasoning_1>...</reasoning_1>
<old_str_1>...</old_str_1>
<new_str_1>...</new_str_1>
```

然后由 `agent/utils.py::extract_edits` 和 `str_replace` 应用修改。

优点：

- 比重新生成完整 code 更节省 token；
- 适合局部修复和小范围性能调优。

问题：

- `str_replace` 必须 exact match 且唯一；
- 若 match 失败，当前只是 warning 并返回原代码，可能造成无效迭代；
- 没有 AST/syntax validation；
- 没有检测 tuned code 是否真的发生变化；
- 对错误类型没有结构化 repair policy。

### 3.3 Search Algorithm

#### Iterative Agent

实现位置：`agent/iterative_agent.py::run_iterative_loop`

流程：

```text
如果没有历史 kernel：
  propose initial kernel
否则：
  根据 previous kernels + metrics 生成 str_replace edits
  应用 edits
  evaluate tuned kernel
  更新 local best
  保留最近 max_memory_round 个历史样本
```

Ranking 由 `agent/eval.py::calculate_score` 决定：

```python
not compiled -> (0, 0, 0)
not correct  -> (1, 0, 0)
correct      -> (1, 1, speedup)
```

优点：

- 实现简单；
- 可 resume；
- 每一步都有 prompt/code/metrics artifact。

缺点：

- 单条优化轨迹容易陷入局部最优；
- 最近 memory 会遗忘早期但有价值的 design；
- 错误样本可能污染 prompt；
- 只用 speedup 排序，不考虑 per-workload robustness、variance、profile health、branch diversity。

#### Evolve Agent

实现位置：`agent/evolve_agent.py::run_evolve_loop`

核心状态：

- `kernel_pool`
- `metrics_pool`
- `proposal_ids`
- `elite_pool`
- `elite_metrics_pool`
- `elite_proposal_ids`

每轮：

1. 选取最近 `pool_size // 2` 个 proposal；
2. 从 correct elite 中按 speedup softmax 采样；
3. 将 recent + elite 放入 proposer prompt；
4. 生成新 proposal；
5. 评估并加入 pool；
6. 按 `calculate_score` 排序更新 elite pool。

优点：

- 比 iterative 更有探索能力；
- recent + elite context 简单有效；
- 支持 softmax temperature 调整探索/利用。

缺点：

- 还不是真正的 evolutionary search：没有 mutation operator、crossover、descriptor、novelty、branch lineage；
- elite 只按 speedup 采样，没有多目标/Pareto；
- 不做 code hash dedup；
- 不区分不同优化 family，例如 tile strategy、layout strategy、CUDA vs Triton path。

## 4. Performance Model 与 Evaluator

### 4.1 当前 EvalResult

`agent/eval.py` 定义：

```python
class EvalResult(BaseModel):
    compiled: bool = False
    correct: bool = False
    speedup: float = 0.0
    latency_ms: float | None = None
    task_id: str = ""
    error: str | None = None
    stats: dict | None = None
```

### 4.2 本地评估流程

`eval_kernel()` 做的事情：

1. `TraceSet.from_path(dataset_root)` 加载 dataset；
2. 创建临时 `Solution`；
3. 设置 `BuildSpec`：
   - `language=TRITON` 默认；
   - `target_hardware=["cuda"]`；
   - `entry_point="main.py::run"`；
   - `destination_passing_style=False`；
4. 把 solution 注入 trace set；
5. 用 `BenchmarkConfig` 跑 benchmark：
   - `warmup_runs=3`；
   - `iterations=5`；
   - `num_trials=1`；
   - `timeout_seconds=60`；
6. 遇到 compile/runtime/correctness/timeout error 即返回失败；
7. 对 passed traces 求平均 speedup 和 latency。

### 4.3 当前 performance model 的限制

当前所谓 performance model 实际上是 **measurement-only model**：只使用 FlashInfer-Bench 实测结果，没有预测模型或代价模型。

缺少：

- 编译失败概率预测；
- correctness failure 分类；
- per-workload speedup 分布；
- latency variance / confidence interval；
- profiler metrics；
- occupancy/register/shared-memory/memory-bandwidth 信息；
- official score estimator；
- 对 full benchmark 前的 cheap surrogate ranking。

这会导致搜索成本较高：每个 candidate 都可能直接进入较重的 benchmark，无法在早期用低成本信号筛掉明显无效的方向。

## 5. Testing Framework 与官方评测差异

### 5.1 Starter Kit 测试流程

`flashinfer-bench-starter-kit/scripts/run_local.py`：

```text
pack_solution()
  -> Solution.model_validate_json
  -> TraceSet.from_path(FIB_DATASET_PATH)
  -> 构造只包含目标 definition/workloads/solution 的 TraceSet
  -> Benchmark.run_all(dump_traces=True)
  -> 打印每个 workload 的 status/latency/speedup/error
```

`run_modal.py` 逻辑类似，只是放到 Modal B200 上执行。

### 5.2 官方评测环境

`EVALUATION.md` 中官方环境：

| 项 | 值 |
|---|---|
| Hardware | Bare-metal NVIDIA B200, `sm_100a` |
| CUDA | 13.2 |
| Python | 3.12 |
| PyTorch | 2.12.0+cu132 |
| Triton | 3.6.0 |
| GPU clocks | locked to max |
| Runner | isolated subprocess |
| Timeout | 300s |

官方 scoring：

1. 每个 kernel 的 speedup 是 `FlashInfer baseline latency / candidate latency` 的 arithmetic mean；
2. 任意 workload fail，则该 kernel 分数为 0；
3. DSA 和 GDN 是 multi-kernel track，缺一个 definition 等价于该 track 被扣一半。

### 5.3 当前 agent 与官方评测的差异

| 维度 | 当前 agent | 官方评测 |
|---|---|---|
| timeout | 默认 60s | 300s |
| trials | `num_trials=1` | 官方命令更完整，部分任务有多 trial |
| tolerances | 通用 FlashInfer-Bench 默认 | MoE 有特殊 `--atol 1 --rtol 0.3 --required-matched-ratio 0.9` |
| scoring | per-definition average speedup | per-track score，missing kernel 计 0 |
| runner | local/modal Python wrapper | isolated subprocess |
| profiling | 默认无 | 环境支持 NCU/CUPTI，但是否用于最终打分取决于评测流程 |

因此，agent 的 local best 不一定等价于 official best。需要引入 official-score-aware evaluation mode。

## 6. 组件交互流程

当前组件交互可以总结为：

```text
Dataset definitions/workloads/baseline solutions
        |
        v
utils.load_tasks_from_test_list / load_test_source
        |
        v
main.run_agent 构造 task_params
        |
        v
proposer_prompt / tuner_prompt
        |
        v
api.query_inference_server 调用 LLM
        |
        v
extract_first_code 或 extract_edits + str_replace
        |
        v
eval.eval_kernel / modal_eval.remote_eval_kernel
        |
        v
FlashInfer-Bench Benchmark.run_all
        |
        v
EvalResult(compiled, correct, speedup, latency, error, stats)
        |
        v
iterative local best 或 evolve elite pool 更新
        |
        v
保存 prompt / code / metrics / global best
```

这个流程的优点是闭环清楚、易运行。主要缺少的是：**从评估结果到下一步搜索策略之间的结构化诊断层**。当前 LLM 主要看到 raw code 和 metrics，而不是一个明确的 profile/failure/action plan。

## 7. 主要缺口与瓶颈

### 7.1 Kernel Generator 缺口

| 缺口 | 证据 | 影响 |
|---|---|---|
| 过于 Triton-first | `eval_kernel()` 默认 `backend="triton"`，proposer prompt 明确要求 Triton。 | 可能错过 CUDA/CuTe/DeepGEMM 在 B200 上的优势。 |
| 缺少 op-specific 策略 | prompt 直接塞 definition JSON。 | MoE/GDN/DSA 复杂结构难靠通用 prompt 自动发现强实现。 |
| 缺少模板库 | starter kit 中 Triton/CUDA 模板都是 TODO。 | 每次都从零生成 boilerplate 和 launch logic。 |
| device handling 指令可能过重 | prompt 要求 CPU/GPU tensor movement。 | 可能影响 benchmark 语义和性能。 |
| 无静态检查 | 生成代码直接进入 benchmark。 | GPU 评估时间被语法/import/signature 错误浪费。 |

### 7.2 Search Algorithm 缺口

| 缺口 | 影响 |
|---|---|
| iterative 是单轨迹搜索 | 容易陷入错误架构或局部最优。 |
| `str_replace` brittle | 匹配失败会造成无效 tuning step。 |
| evolve 没有 branch descriptor | diversity 不可控，candidate 容易同质化。 |
| 排序只有 `(compiled, correct, speedup)` | 忽略 variance、tail workload、profile health、compile cost。 |
| 无 code hash dedup/cache | 重复 candidate 会浪费编译和 benchmark 时间。 |

### 7.3 Performance Model 缺口

| 缺口 | 影响 |
|---|---|
| 没有 surrogate model | 无法低成本预测哪些 candidate 值得 benchmark。 |
| 没有 profiler feedback | LLM 不知道瓶颈是 memory、compute、register 还是 occupancy。 |
| 没有 per-workload 聚类 | 可能平均速度好但 tail workload 很差。 |
| `num_trials=1` | 排名容易受噪声影响。 |
| 没有 official score tracker | 搜索目标与最终得分可能不一致。 |

### 7.4 Testing Framework 缺口

| 缺口 | 影响 |
|---|---|
| 没有 staged validation | 无效 kernel 直接进入完整 evaluator。 |
| 没有 sanitizer | race/memory bug 可能后期才暴露。 |
| 没有 NCU profile | 无法解释性能瓶颈。 |
| `EvalResult.error` 是单字符串 | repair prompt 缺少结构化 failure reason。 |
| 没有 hidden/robustness 策略 | 存在对公开 workload 过拟合风险。 |

### 7.5 Artifact / Memory 缺口

| 缺口 | 影响 |
|---|---|
| 没有 `candidates.jsonl` | 难以跨 run 分析 candidate lineage。 |
| 没有 branch graph | 难判断哪些策略 family 有效。 |
| 没有 env snapshot | 难复现实验环境。 |
| 没有 per-workload table | 难定位某些 shape 的退化。 |
| 没有自动 `solution.json` 输出 | 从 best kernel 到提交包还需要人工转换。 |

## 8. 改进建议

### 8.1 增加 TaskProfile 和任务专用 skill

在 prompt 前先把 definition 转成结构化 TaskProfile：

```yaml
definition: gdn_decode_qk4_v8_d128_k_last
op_type: gdn
stage: decode
heads:
  q: 4
  k: 4
  v: 8
head_size: 128
state_layout: k-last
primary_dtypes:
  qkv: bfloat16
  state: float32
candidate_strategies:
  - fused_gate_state_update
  - batch_head_parallel
  - state_tile_reuse
```

建议建立 skill packs：

| Skill | 内容 |
|---|---|
| `moe-fp8-blackwell` | routing、top-k、FP8 block scale、grouped GEMM、DeepGEMM/CuTe strategy。 |
| `gdn-recurrent-state` | decode/prefill 区分、k-last layout、state update fusion、BF16/FP32 accumulation。 |
| `dsa-paged-attention` | paged KV cache、top-k sparse index、online softmax、KPE handling。 |
| `blackwell-b200-kernel` | SM100/B200 memory hierarchy、occupancy/register tradeoff、TMA/WGMMA/CuTe hints。 |

### 8.2 引入多分支/Pareto 搜索

为 candidate 增加 metadata：

```yaml
candidate_id: c0012
parent_id: c0007
branch_family: triton_tiled_gdn_decode
language: triton
strategy: greedy_refine | repair | evolve | template_mutation
mutation_type: tile_shape | memory_layout | vectorization | algorithm
compiled: true
correct: true
speedup: 1.18
per_workload_speedups: [...]
profile_summary_ref: profile/c0012.json
```

搜索池不应只按 speedup 排序，而应保留：

- top speedup candidates；
- per-branch best；
- Pareto frontier；
- near-best but structurally diverse candidates；
- 典型 failure repair examples。

### 8.3 分层评估 pipeline

建议将 evaluation 拆成：

```text
static syntax/import check
  -> signature/DPS check
  -> compile smoke test
  -> correctness subset
  -> quick benchmark subset
  -> full workload benchmark
  -> NCU/sanitizer for finalists
  -> official score estimate
```

这样可以减少无效 GPU benchmark，提高搜索吞吐。

### 8.4 加入 profiler-guided feedback

Starter kit 已经展示了：

- `flashinfer_bench_run_ncu`
- `flashinfer_bench_run_sanitizer`

应将其接入 agent：

| Profile 现象 | 下一步策略 |
|---|---|
| occupancy 低 | 降低 register pressure、减少 live tensor、调整 block size。 |
| memory bandwidth 饱和 | 改善 coalescing、vectorized load、reuse。 |
| shared memory conflict | 修改 layout/padding。 |
| tensor core 利用低 | 切换 MMA/CuTe/DeepGEMM strategy。 |
| latency variance 大 | 稳定 launch config，减少动态分支。 |

### 8.5 对齐官方 scoring

增加 official score tracker：

- per-kernel score；
- per-track score；
- DSA/GDN missing kernel penalty；
- 与 FlashInfer baseline 的 speedup；
- final submission readiness。

这样 agent 才能按最终比赛目标分配预算。例如 GDN 不应只优化 decode，也要考虑 prefill，否则 track score 被缺失 definition 拉低。

### 8.6 支持 CUDA/CuTe/DeepGEMM 路径

B200/SM100 上很多高性能 kernel 可能需要低层控制。建议添加：

- `language_strategy: triton | cuda | cute_dsl | deep_gemm_wrapper`；
- CUDA binding generation prompt；
- TVM FFI / torch binding 模板；
- CUDA compile-only stage；
- MoE FP8 GEMM 的 DeepGEMM/CuTe skeleton；
- DSA/GDN 的 CUDA persistent/tiled skeleton。

### 8.7 改进 `str_replace` patch 机制

建议：

- no-op edit 直接标记失败；
- edit 后做 `ast.parse()`；
- 记录 patch diff；
- 对重复失败回退到 full regeneration；
- 支持 unified diff 或 AST-aware patch；
- 对每个 edit 记录 reason、applied、changed_lines。

### 8.8 增加 caching 和 dedup

按以下 key 缓存：

- source code hash；
- definition id；
- benchmark config；
- dataset hash；
- hardware / backend / language；
- dependency version。

避免重复编译和重复 benchmark 相同 kernel。

### 8.9 改进 artifact schema

推荐输出结构：

```text
outputs/<run>/
  config.yaml
  env.json
  run_summary.md
  candidates.jsonl
  branches.jsonl
  scoreboard.csv
  <op>_<definition>/
    definition.json
    reference_src.py
    candidate_0001.py
    candidate_0001_metrics.json
    candidate_0001_profile.json
    candidate_0001_prompt.txt
    best_solution.json
```

最终报告应包含：

- task summary；
- candidate timeline；
- per-workload speedup；
- best kernel lineage；
- failure taxonomy；
- profiler observations；
- official score estimate；
- next recommended search branch。

## 9. 推荐的新架构

```text
Track / Task Planner
  -> 选择 op/definition/预算
  -> 加载 definition、workloads、baseline

TaskProfile Builder
  -> 解析 shape、dtype、layout、constraints
  -> 选择 op-specific skill

Strategy Selector
  -> Triton / CUDA / CuTe / DeepGEMM
  -> repair / refine / evolve / template mutation

Kernel Generator
  -> 生成或修改 candidate
  -> 附带 branch metadata

Tiered Evaluator
  -> syntax/import check
  -> compile smoke
  -> correctness subset
  -> quick benchmark
  -> full benchmark
  -> NCU/sanitizer for finalists

Evidence Store
  -> candidates.jsonl
  -> per-workload metrics
  -> branch lineage
  -> profile summary
  -> official score estimate

Search Controller
  -> 更新 elite/diverse/Pareto pools
  -> 分配下一轮预算
  -> promote / reject / continue
```

核心原则：

> LLM 负责提出优化假设和代码变换；FlashInfer-Bench、sanitizer、profiler 负责决定事实。

## 10. 分阶段实施路线

### Phase 0：Baseline hardening

- 加入 Python AST / import / signature precheck。
- `str_replace` no-op 视为失败并记录。
- 增加 code hash dedup。
- 输出 `candidates.jsonl`。
- 增加 per-workload metrics。

### Phase 1：Evaluation alignment

- 对齐官方 timeout/tolerance。
- 增加 official score tracker。
- 引入 quick/full benchmark modes。
- 对 DSA/GDN 做 track-level budget planning。

### Phase 2：Search improvement

- 增加 branch family metadata。
- 实现 diversity-aware elite selection。
- 实现 Pareto frontier 和 top-K finalist selection。
- 将 failure taxonomy 作为 repair prompt 输入。

### Phase 3：Profiler-guided optimization

- 对 finalists 跑 NCU summary。
- 对可疑 candidates 跑 sanitizer。
- 建立 profile-to-action 映射。
- 将 profiler facts 压缩进 tuner prompt。

### Phase 4：Advanced backend support

- CUDA/TVM FFI 生成路径一等支持。
- CuTe/CUTLASS DSL skeleton。
- DeepGEMM wrapper strategy。
- 自动打包 `solution.json`。

## 11. 结论

MLSys26 agent baseline 是一个清晰、可运行、适合作为起点的 LLM kernel generation baseline。它已经具备：

- LLM code generation；
- iterative refinement；
- evolve-style pool search；
- FlashInfer-Bench correctness/performance evaluation；
- local 和 Modal/B200 backend；
- 基础 artifact 保存和 resume。

但如果目标是构建有竞争力的 MLSys26 kernel optimization agent，仅靠当前 prompt + benchmark loop 不够。下一步应重点补齐：

1. task-aware planning；
2. MoE/GDN/DSA/Blackwell skill；
3. multi-branch/Pareto search；
4. staged evaluation；
5. profiler-guided optimization；
6. official scoring alignment；
7. CUDA/CuTe/DeepGEMM backend；
8. richer evidence and memory。

这些增强可以在不破坏现有 baseline 的基础上逐步加入，使系统从“能自动试 kernel”升级为“能基于证据系统性优化 kernel”的 agent framework。
