# Kernel Foundry 框架分析

> 日期：2026-06-01  
> 范围：`/mnt/river/kernel_foundry/kernelfoundry.internal`、`/mnt/river/kernel_foundry/kernelfoundry.kernel-eval`、`/mnt/river/kernel_foundry/kernelfoundry.templates`  
> 目标：理解当前 Kernel Foundry 架构，识别瓶颈/缺口，并为 kernel optimization agent 提出改进建议。

## 执行摘要

Kernel Foundry 已经是一个较完整的 kernel 生成与评估平台。它结合了基于 LLM 的代码生成、任务打包、构建/测试/性能分析执行、RAG 示例、数据库日志、可选的 MAP-Elites 风格进化，以及面向 SYCL/CUDA/Triton/OpenCL GPU kernel 的 benchmark/任务模板。

最强的可复用资产包括：

- `CustomTask` 任务抽象：带有构建/测试/性能分析元数据的 `REFERENCE`、`EVOLVE` 和 `USER_INSTRUCTIONS` 标注代码块。
- `CustomTaskController`：带反馈、评估、数据库日志和可选进化的迭代式 LLM 生成循环。
- `CustomTaskEvaluator` + `TaskRunner`：确定性的构建、正确性、性能与 profiler 执行流水线。
- Profiler 反馈：基于 unitrace 的 Intel GPU 分析，以及面向 CUDA 的 NCU 支持。
- RAG 与程序数据库：持久化 kernel/result 存储与检索。
- 模板：PyTorch→SYCL、SYCL→SYCL、OpenCL→OpenCL、模型层任务和 OpenCL PyTorch operation 的实用任务模式。

不过，当前系统仍主要是一个批量实验/搜索 harness。它还不是面向用户的端到端优化 agent。主要缺口包括：任务/配置复杂度高、单次 trial 成本高、benchmark 方向偏静态、runtime 集成有限、anti-reward-hacking 保障较弱，以及搜索策略更多由框架驱动而非 workload 驱动。

## 仓库组件

### 1. `kernelfoundry.internal`

这是主要的生成与编排仓库。

重要模块：

| 组件 | 路径 | 职责 |
|---|---|---|
| 主控制器 | `kernelgen/controller.py` | Prompt 构造、LLM 调用、parent/inspiration 采样、optimization-aware prompting。 |
| Custom task 控制器 | `kernelgen/custom_task_controller.py` | Custom task 的 trial 循环：生成、抽取、评估、记录、选择最佳、提前停止。 |
| 评估器 | `kernelgen/custom_task_evaluator.py` | 将 build/test/profile 输出转换为带 correctness、runtime、speedup、profiler data 的 `EvalResult`。 |
| 任务抽象 | `kernelgen/custom_task.py` | 内存态任务归档、代码块抽取/更新、测试名、任务配置、DB 序列化。 |
| 任务分发器 | `kernelgen/tasks/task_runner.py` | 本地或 Celery 的构建/测试/image 执行。 |
| Runtime 测试任务 | `kernelgen/tasks/test_custom_task.py` | 运行 pytest correctness、pytest performance 和 profiler passes。 |
| LLM server | `kernelgen/inference_server.py` | OpenAI-compatible、AWS Bedrock、GNAI、IBM、本地 Llama Factory、ensembles。 |
| Prompt 构造 | `kernelgen/prompts/prompt_constructor.py` | 基于模板的 prompt 拼装、RAG 检索、代码分类。 |
| Optimization-aware prompting | `kernelgen/prompts/optimization_aware.py` | 面向 backend 的 memory/compute/parallelism/ESIMD 指导。 |
| Pattern 分类器 | `kernelgen/utils/map_elites_patterns.py` | 面向 SYCL/CUDA/OCL/ESIMD 的正则行为画像检测。 |
| QD gradient | `kernelgen/qd_gradient.py` | MAP-Elites 进化的 transition tracking 和类梯度信号。 |
| Profiler 反馈 | `kernelgen/profiler_feedback.py` | unitrace 和 NCU 指标整理与反馈文本。 |
| 数据库 schema | `kernelgen/database/tables.py` | `Kernel`、`Task`、`Job`、`JobLog`、`Rag`、baseline timing 表。 |
| MCP tool server | `kernelgen/mcp/server.py` | 用于 server-side validation 的 `build_and_test(folder_path)` 封装。 |
| 核心编译器 | `kernelfoundry/kernelfoundry/compiler.py` | `TorchCompiler` 和 `IcpxCompiler` 构建辅助。 |

### 2. `kernelfoundry.kernel-eval`

这是面向 Intel GPU kernel 生成任务的 benchmark/evaluation package。

它定义了：

- 任务组：oneDNN、oneDAL、OpenVINO/OpenCL。
- 标准任务结构：`task.py`、`conftest.py`、`config.yaml`、`*_kernel.*`、`*_reference.*`，以及可选 build folders/scripts/CMake 文件。
- 基于 Python pytest 的验证和 benchmark harness。
- 示例：`tasks/ov_ocl/gated_delta_net/task.py` 使用 PyOpenCL、随机输入、OpenCL kernel 编译、正确性检查以及基于 `measure_runtime` 的 profiling。

该 benchmark 仓库很有价值，因为它将任务定义与生成逻辑解耦。它可以作为新 agent 的评估 backend 复用。

### 3. `kernelfoundry.templates`

该仓库包含可复用的任务模板。

支持的用例：

- PyTorch operation 到 SYCL kernel。
- SYCL kernel 优化。
- 面向模型层的 SYCL kernel。
- PyTorch operation 到 OpenCL kernel。
- OpenCL kernel 优化。

模板模型对于新任务 onboarding 很重要，因为它为 `REFERENCE`、`EVOLVE`、测试设计和配置提供了具体模式。

## 当前端到端流程

当前 custom task 生成循环大致如下：

1. 用户提供 Hydra 配置和 custom task 路径。
2. `CustomTask.create()` 加载任务文件，抽取代码块、测试名和配置。
3. `CustomTaskController` 初始化 LLM、prompt constructor、feedback helper 和可选 program database。
4. 对每个 trial：
   - 基于 reference code、上一轮 program/eval feedback、RAG 示例和优化指导构造 prompt。
   - 查询 LLM server 生成一个或多个候选实现。
   - 将生成代码抽取到 `EVOLVE` block。
   - 可选执行 syntax precheck。
   - 用 `CustomTaskEvaluator` 评估每个候选。
   - 通过 `TaskRunner` 运行 build、correctness tests、performance tests 和 profiler passes。
   - 将结果转换为带 `compiled`、`correctness`、`perf_score`、`runtime`、`runtime_improvement` 和 profiler metadata 的 `EvalResult`。
   - 将生成 kernel/result/logs 存入 run directory 和可选数据库。
   - 选择最佳结果，并将其作为 parent 或加入 program database。
5. 在达到最大迭代次数、正确性目标、提前停滞或失败限制时停止。

这是一个健壮的实验循环，但它仍然假设任务、backend、指标和评估目标大多已经预配置。

## 架构优势

### 1. 清晰的任务抽象

`CustomTask` 是一个很强的边界。它将任务数据存储在 memory file map 中，检测 build/test/profile 步骤，抽取 `REFERENCE` 和 `EVOLVE` 区域，并能在不修改无关文件的情况下更新代码块。这正是 optimization agent 所需要的确定性环境接口。

### 2. 确定性评估器

`CustomTaskEvaluator` 区分了：

- 代码抽取错误，
- 构建失败，
- runtime/test 失败，
- 正确性失败，
- 性能测量，
- profiler 反馈，
- speedup 计算。

这种结构化结果适合 agent feedback 和 reward 设计。

### 3. 多语言 kernel 支持

当前支持包括：

- SYCL，
- CUDA，
- Triton，
- OpenCL。

构建支持更窄一些（`cuda`、`sycl`、`ocl`），但 prompt 和 profiling 逻辑已经理解多个 backend。

### 4. Intel GPU profiling 支持

`UnitraceProfilerFeedback` 会抽取 runtime、occupancy、memory bandwidth、SLM usage/conflicts、XVE stalls、ALU utilization，以及 roofline 风格的 bound classification。这是面向 Intel GPU kernel 优化的重要差异化能力。

### 5. RAG 与历史数据库

`Kernel`、`Task`、`Job` 和 `Rag` 表为以下能力提供基础：

- 从历史 kernel warm-start，
- 检索相关示例，
- 跟踪 model/config/profiler metadata，
- 跨时间比较性能。

### 6. 进化基础设施

代码中已经包含：

- 行为维度：memory、compute、parallelism、ESIMD，
- 基于正则的 optimization profile detection，
- 类 MAP-Elites program database，
- 类梯度 transition tracking，
- optimization-aware prompt mutation。

这些可以作为更大 agent 中的一种搜索策略复用。

### 7. MCP 入口

`kernelgen/mcp/server.py` 暴露了 `build_and_test(folder_path)`，这是 coding agent 的有用 tool 边界：agent 可以生成/编辑任务，然后请求 Kernel Foundry 验证。

## 主要缺口与瓶颈

### 1. 任务/配置复杂度

默认 workflow 依赖大量 Hydra 选项：

- `job_name`、`task_name`、`custom_task`、`task_origin`、`language`、`gpu_arch`、`mode`、`evolve_mode`、queue settings、timeouts、profile settings、database settings、prompt settings、container settings。

这很强大，但对于只想“在这个 GPU 上优化这个 kernel/workload”的用户来说很困难。新的 agent 应将自然语言转换为紧凑的 task spec，然后自动生成 Hydra/custom-task 细节。

### 2. 单次 trial 延迟高

一次完整 candidate evaluation 可能包含：

- LLM inference，
- build，
- correctness pytest，
- performance pytest，
- 多个 profiler passes，
- 可选 queue/container overhead。

`TaskRunner.test_custom_task()` 会按 profiler/correctness/performance 子进程放大 test timeout。这对可靠评估是合理的，但对内层探索很昂贵。

框架需要分层 evaluator：

1. 快速 syntax/static checks，
2. 最小 correctness smoke tests，
3. 在少量代表性 shape 上做 microbenchmark，
4. 只对有希望的候选做完整 correctness/performance/profile，
5. 只对 finalist 做 E2E workload benchmark。

### 3. Token 成本与 prompt 膨胀

Prompt 构造可能包含：

- 完整 reference source，
- vector-add 示例，
- RAG 示例，
- previous program，
- top programs，
- inspirations，
- profiler logs，
- feedback LLM output，
- optimization taxonomy/guidance。

这些有用但可能昂贵。设计应推动：

- 更小的 delta prompts，
- 缓存静态 skills，
- 用结构化 error summaries 替代 raw logs，
- 从数据库示例 warm-start，
- 显式 token budgets。

### 4. 批量搜索导向

当前 controller 关注在已配置任务上生成候选。它还没有拥有完整循环：

- workload profiling，
- hotspot selection，
- kernel target extraction，
- runtime integration，
- E2E benchmarking，
- rollout/rollback decisions。

真实优化中，agent 必须优化 workload，而不只是孤立任务。

### 5. MAP-Elites 应作为策略，而非 orchestrator

当前 evolve mode 可以从 program database 中采样 parents/inspirations 并使用 behavior descriptors。这对多样性有价值，但对许多 kernel task 来说，聚焦的 profile-driven refine loop 更高效。

建议变化：将 MAP-Elites/gradient tracking 保留为可选 `search_strategy`，仅在以下情况调用：

- 搜索空间很宽，
- greedy refinement 停滞，
- 多种 kernel family 都可能可行，
- 多样性对跳出局部最优有价值。

### 6. Anti-reward-hacking 保障较弱

当前 evaluator 将正确性交给 task-defined pytest tests。有些任务很健壮，但框架本身没有强制：

- randomized hidden shapes，
- output buffer poisoning，
- 与 previous reference outputs 隔离，
- 基于 profiler 的“是否真的做了计算？”检查，
- runtime 集成后的 E2E output equivalence。

带 reward feedback 的 optimization agent 需要更强的默认验证，以避免 overfitting 或 gaming tests。

### 7. Runtime 集成不是一等阶段

Kernel Foundry 能生成和验证任务 kernel，但还没有提供通用 `integrate_runtime` 抽象，用于：

- OpenVINO custom op/extension，
- PyTorch-XPU custom op，
- oneDNN primitive replacement，
- OpenCL runtime kernel replacement，
- rollback 和 provenance。

这是实现真正端到端 workload optimization 的最大缺口。

### 8. Profiling 是 candidate-level，而非 workload-level

Profiler feedback 在 candidate 运行后很有用，但当前没有高层 `profile_workload` 阶段，用 E2E contribution 决定应优先优化哪个 op/kernel。

### 9. 测试用例可能是静态的

模板任务经常在 `task.py` 中硬编码 shapes。这对 benchmark 有用，但生成的 kernel 可能 overfit 到这些 shapes。新 agent 应支持 shape distribution specs 和动态 test generation。

## 改进建议

### P0：将 Kernel Foundry 变成可调用的确定性 tools

在现有组件之上定义稳定 tool interfaces：

- `load_task(path) -> TaskSpec`
- `build_kernel(task, candidate) -> BuildResult`
- `verify_correctness(task, candidate, shape_policy) -> VerificationResult`
- `benchmark_micro(task, candidate, shape_policy) -> BenchmarkResult`
- `profile_kernel(task, candidate) -> ProfileResult`
- `query_kernel_memory(signature, hw) -> candidates`
- `record_candidate(...) -> id`

MCP server 已经指向这个方向；应将其从 `build_and_test` 扩展出去。

### P1：引入分层评估

使用多个评估层级：

| 层级 | 目的 | 预期成本 |
|---|---|---|
| T0 static/syntax | 捕获明显的抽取/编译失败 | 秒级 |
| T1 smoke correctness | 一个或两个小型随机 case | 秒级 |
| T2 micro correctness/perf | 代表性 shape set | 数十秒 |
| T3 profiler | finalist 的 unitrace/ncu feedback | 昂贵 |
| T4 E2E | 真实 workload 对比 | 最昂贵 |

这会直接解决单 trial 评估时间过长的问题。

### P2：增加 natural-language-to-task-spec 转换

前端 planner 应将用户意图转换为：

- target runtime，
- target hardware，
- source workload 或 task template，
- correctness tolerance，
- shape distribution，
- metric priority，
- budget，
- allowed languages/backends，
- integration mode。

Hydra config 保持内部实现细节；常见 workflow 中用户不应手写它。

### P3：强化验证

增加默认 anti-gaming checks：

- 多个 randomized seeds，
- hidden/held-out shapes，
- 非 2 的幂维度，
- output buffer poisoning，
- reference/custom buffer isolation，
- profiler sanity check for memory/compute activity，
- 集成后的 E2E output equivalence。

### P4：提升 profiler-driven search 地位

不要只把 `UnitraceProfilerFeedback` 和 `NCUProfilerFeedback` 作为追加到 prompt 的文本，而应作为结构化 strategy signals：

- memory-bound → coalescing、vector loads、SLM/register tiling、减少 transfers，
- compute-bound → DPAS/XMX/tensor core、ILP、unroll、FMA、algorithmic fusion，
- latency/stall-bound → 减少 sync、提升 occupancy、persistent/thread mapping 调整，
- low occupancy → work-group/sub-group size tuning、降低 register pressure，
- SLM conflicts → padding/layout changes。

### P5：使搜索策略可插拔

推荐 strategy API：

- `greedy_refine`：默认低 token 串行循环。
- `beam_search`：维护 top-k candidates。
- `evolution`：使用现有 DB/parent/inspiration sampling。
- `map_elites`：使用 behavior descriptors 做 diversity。
- `fallback_repair`：compile/runtime error repair mode。
- `hyperparameter_tune`：tile/work-group/sub-group 参数扫描。

### P6：扩展数据库 schema 支持 E2E optimization

添加或附加字段：

- `workload_id`，
- `op_signature`，
- `shape_distribution`，
- `runtime_backend`，
- `integration_mode`，
- `e2e_latency_stats`，
- `micro_latency_stats`，
- `strategy_used`，
- `validation_policy`，
- `promotion_status`。

### P7：将知识拆分为 skills

将可复用 prompt knowledge 抽取为紧凑 skill docs/assets：

- Intel GPU architecture specs，
- SYCL/OpenCL optimization idioms，
- ESIMD/DPAS/XMX patterns，
- profiler diagnosis playbooks，
- runtime integration guides，
- task template authoring guide。

这能减少重复 prompt 成本，并让 agent 行为更易维护。

## Kernel Optimization Agent 复用映射

| 现有 KF 资产 | 复用方式 | 说明 |
|---|---|---|
| `CustomTask` | Environment/task representation | 保留。增加 shape policy 和 NL metadata。 |
| `CustomTaskEvaluator` | Evaluation tool backend | 保留。增加 tiered policies 和 anti-gaming。 |
| `TaskRunner` | Local/queue executor | 保留。增加显式 scheduling 和 budget control。 |
| `PromptConstructor` | Candidate generation helper | 选择性复用；优先使用更小的 agent prompts。 |
| `optimization_aware.py` | Strategy/skill source | 抽取为 skills 和结构化 decision rules。 |
| `map_elites_patterns.py` | Behavior descriptor extractor | 保留，用于可选 QD strategy。 |
| `qd_gradient.py` | Search telemetry | 保留，用于可选 MAP-Elites/gradient search。 |
| `UnitraceProfilerFeedback` | Profile diagnosis tool | 提升为 structured tool output。 |
| `database/tables.py` | Memory/experiment DB | 为 E2E 和 strategy metadata 扩展。 |
| `mcp/server.py` | Agent tool boundary | 扩展为完整 build/verify/bench/profile tools。 |
| `kernelfoundry.templates` | Task bootstrap library | 用于 NL-to-task synthesis。 |
| `kernelfoundry.kernel-eval/tasks` | Regression/eval suite | 用作 benchmark 和 examples。 |

## 优先路线图

### Phase 1：Tool 稳定化

- 将 MCP/API 从 `build_and_test` 扩展为独立的 `build`、`verify`、`benchmark`、`profile`、`record` tools。
- 返回 structured JSON，而不仅是 eval logs。
- 增加确定性 local mode 以支持快速 PoC。

### Phase 2：分层与强化评估

- 实现 smoke/full/hidden shape policies。
- 增加 output poisoning 和 reference/custom isolation helpers。
- 增加 profiler sanity checks。

### Phase 3：Agent loop

- 实现 natural-language task parser。
- 增加 profile→select→generate→build→verify→benchmark→record 循环。
- 默认使用带 compile/runtime feedback 的 greedy refine。

### Phase 4：Runtime 集成

- 从 PyTorch-XPU 或 OpenCL task integration 开始，以便快速 PoC。
- 添加 OpenVINO extension path，以提升生产价值。
- 增加 rollback 和 E2E validation。

### Phase 5：Search 与 memory

- 从 DB/RAG 增加 warm-start retrieval。
- 仅将 MAP-Elites 作为可选 exploration strategy。
- 按 op type、bound type、hardware 和 shape distribution 跟踪 strategy success。

## 结论

Kernel Foundry 已经拥有 kernel optimization 所需的大部分硬基础设施：build/test/profile、task templates、RAG、database 和 evolution logic。下一步是架构性的：将这些组件封装为 agentic orchestrator 周围的确定性 tools，使其能从自然语言需求和真实 workload profiling 出发，以强验证和 E2E promotion criteria 执行 evidence-based closed-loop optimization。
