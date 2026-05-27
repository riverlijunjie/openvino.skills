# KernelFoundry 完整开发指南

> 来源: [Intel Wiki](https://wiki.ith.intel.com/spaces/VCL/pages/4501743035/KernelFoundry) + 仓库文档

## 1. 项目概述

**KernelFoundry** 是 Intel Client AI Solutions & Planning 团队开发的 AI 驱动的 GPU Kernel 代码生成与优化平台。它利用 LLM（大型语言模型）通过进化算法自动生成、调试和优化计算 kernel，将传统需要数天/数周的 kernel 开发缩短到数小时。

### 核心能力
- **自动化 Kernel 生成**: 接受任务定义和测试规范，LLM 自动生成满足要求的 kernel
- **进化式优化**: 通过迭代的生成-测试-反馈循环不断改进 kernel 性能
- **多语言支持**: SYCL、CUDA、Triton、OpenCL
- **多硬件架构**: Intel (PTL/LNL/BMG/DG2) 和 NVIDIA (Ampere)
- **双重访问方式**: Web UI + VS Code 扩展
- **模板驱动开发**: 预置常见 use case 模板
- **内置质量保证**: 自动化正确性测试 + 性能 benchmark
- **RAG 知识增强**: 基于已有高性能 kernel 数据库的检索增强
- **Profiler 反馈**: 基于 unitrace 的性能分析指导优化
- **分布式执行**: Celery + RabbitMQ 任务队列

### 目标用户
性能工程师、ML 研究员和需要优化 AI/ML 工作负载、科学计算或高性能计算的 kernel 的开发者。

### 仓库结构

| 仓库 | 用途 |
|------|------|
| `kernelfoundry.internal` | 主仓库：LLM pipeline、推理、评估、数据库、prompt 工程 |
| `kernelfoundry.kernel-eval` | Benchmark 评估套件：标准化 task 定义 + 评估工具（`kernelfoundry` 包） |
| `kernelfoundry.templates` | 用户模板：各种 use case 的 task 模板和 config 模板 |

---

## 2. 环境安装

### 2.1 Intel 硬件

```bash
# 安装/验证 oneAPI 驱动
./scripts/setup/setup_oneapi.sh --install --oneapi 2025.1 --device A770

# 设置 Python 环境
./scripts/setup/setup_repo.sh --platform intel --machine local
# → 创建 conda 环境: kernel-intel-py312

# 手动安装方式
conda create --name kernel_intel python=3.12.9
conda activate kernel_intel
conda install libclang=20
pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/xpu
pip install -r requirements.txt
source /opt/intel/oneapi/2025.0/oneapi-vars.sh
```

**验证的兼容组合**:
- ✅ oneAPI 2025.0 + PyTorch 2.7.0
- ✅ oneAPI 2025.1 + PyTorch 2.8.0 (默认)

### 2.2 NVIDIA 硬件

```bash
./scripts/setup/setup_repo.sh --platform cuda --machine cluster

# 或手动：
conda create --name kernel_cuda python=3.12.9
conda activate kernel_cuda
conda install libclang=20
pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu129
pip install -r requirements.txt
```

### 2.3 Unitrace (Intel GPU Profiling)

```bash
git clone https://github.com/intel/pti-gpu.git
cd pti-gpu/tools/unitrace && mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_WITH_MPI=OFF ..
make
ln -s $PWD/unitrace $CONDA_PREFIX/bin/unitrace

# 每次重启后需执行：
sudo sh -c 'echo 0 > /proc/sys/dev/i915/perf_stream_paranoid'
sudo sh -c 'echo 0 > /proc/sys/dev/xe/observation_paranoid'
```

### 2.4 LLM API 配置

```bash
# GNAI (Intel 内部):
export GNAI_TOKEN="your_gnai_token"

# Denvr:
export BASE_URL="https://api.inference.denvrdata.com/v1/"
export OPENAI_API_KEY="denvr_api_key_here"

# IBM:
export IBM_API_TOKEN=ibm_token_here
```

获取 GNAI Token: https://gfx-assets.intel.com/auth/oauth2/sso

### 2.5 推荐 `.env` 文件

```
REQUESTS_CA_BUNDLE="/etc/ssl/certs/ca-certificates.crt"
SSL_CERT_FILE="/etc/ssl/certs/ca-certificates.crt"
GNAI_TOKEN="your_gnai_token"
DB_READONLY_PASSWORD="password"
DB_INSERTONLY_PASSWORD="password"
RABBITMQ_IP="isl-igpu1.rr.intel.com"
RABBITMQ_USERNAME="codegen"
RABBITMQ_PASSWORD="password"
QUEUE_BACKEND_TYPE="postgresql"
QUEUE_BACKEND_IP="isl-igpu1.rr.intel.com"
QUEUE_BACKEND_USERNAME="codegen_queue_metadata"
QUEUE_BACKEND_PASSWORD="password"
```

---

## 3. 核心架构

### 3.1 Pipeline 流程

```
Task (PyTorch op) → Prompt Construction → LLM Inference → Code Extraction
    → Compilation → Correctness Test → Performance Benchmark
    → Feedback → (Loop until max_iters or stop_once_correct)
```

### 3.2 核心模块

| 模块 | 文件 | 职责 |
|------|------|------|
| Controller | `kernelgen/controller.py` | 主控制器：prompt构建 + 推理 + 评估循环 |
| CustomTaskController | `kernelgen/custom_task_controller.py` | 自定义 task 控制器 |
| CustomTaskEvaluator | `kernelgen/custom_task_evaluator.py` | 编译 + 测试 + 性能评估 |
| InferenceServer | `kernelgen/inference_server.py` | LLM API 抽象（GNAI/Denvr/IBM/本地） |
| PromptConstructor | `kernelgen/prompts/prompt_constructor.py` | Prompt 模板与上下文组装 |
| FeedbackHelper | `kernelgen/prompts/feedback_llm.py` | 用另一个 LLM 分析评估日志 |
| AnswerProcessor | `kernelgen/answer_processor.py` | 从 LLM 输出中提取代码 |
| ProfilerFeedback | `kernelgen/profiler_feedback.py` | Profiler 数据分析与反馈生成 |
| EvolveDatabase | `kernelgen/evolve_database_optimization_aware.py` | MAP-Elites 进化数据库 |
| TaskRunner | `kernelgen/tasks/task_runner.py` | 通过 Celery 队列分发任务 |
| CeleryApp | `kernelgen/celery_app.py` | 消息队列配置（RabbitMQ + PostgreSQL） |
| RAG | `kernelgen/database/rag_*.py` | 检索增强生成 |
| CustomTask | `kernelgen/custom_task.py` | 自定义 task 的数据结构 |
| Schemas | `kernelgen/schemas.py` | EvalResult, Program 等核心数据模型 |

### 3.3 支持的 LLM 模型

通过 `intel_gnai` 服务提供（模型列表经常更新，查看 https://gpusw-docs.intel.com/services/gnai/models/ ）：

**当前可用**:
- GPT 系列: `gpt-4.1`, `gpt-4o`, `gpt-5-mini`, `gpt-5-nano`, `gpt-5.1`, `gpt-5.1-codex`, `gpt-5.1-codex-max`, `gpt-5.2`, `gpt-5.2-codex`
- 推理模型: `o3`, `o3-mini`, `o4-mini`
- Claude 系列: `claude-4-5-sonnet`, `claude-4-5-sonnet-thinking`, `claude-4-5-haiku`, `claude-4-5-haiku-thinking`, `claude-4-5-opus`, `claude-4-5-opus-thinking`
- 其他（Denvr）: Llama-3.3-70B, Qwen2.5-72B, DeepSeek-R1, Mistral, Falcon 等

### 3.4 分布式架构

```
Controller (脚本) → RabbitMQ Broker → Celery Workers (GPU 节点)
                                ↓
                    PostgreSQL (结果存储)
```

Worker 任务类型:
- `build_image`: 构建 Docker 容器镜像
- `pull_image`: 拉取预构建镜像
- `build_custom_task`: 编译 kernel
- `test_custom_task`: 运行测试

---

## 4. 工作流程与核心概念

### 4.1 Task 与 Job 的关系

- **Task（任务）**: 一个具体的 kernel 生成项目，定义了一个需要编写 kernel 的操作
- **Job（作业）**: Task 的一次执行，包含特定的配置参数（迭代次数、模型选择等）

一个 Task 可以启动多个 Job，例如先用便宜模型快速测试，再用强大模型深度优化。

### 4.2 迭代流程

```
┌─────────────────────── 第一轮迭代 ───────────────────────┐
│ Prompt = Reference + User Instructions + RAG Examples    │
│    → LLM → 生成 Kernel → 编译 → 测试 → Benchmark       │
└──────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────── 后续迭代 ───────────────────────┐
│ Prompt = Reference + 上一轮 Kernel + Eval Log +         │
│          RAG + Best Kernel + Profiler Feedback          │
│    → LLM → 生成改进 Kernel → 编译 → 测试 → Benchmark  │
└─────────────────────────────────────────────────────────┘
                    ↓ (重复 max_iters 次)
```

### 4.3 标准工作流

1. **定义 Task**: 适配模板或从头编写测试和参考实现
2. **验证 Task**: 通过 VS Code 扩展 / Web UI 验证，或本地 `pytest --ref task.py`
3. **运行 Pipeline**: 提交 job 进行 kernel 生成
4. **检查结果**: 查看生成的 kernels 和评估日志

### 4.4 运行方式

**主生成 Pipeline (KernelBench 任务)**:
```bash
# 快速测试 (Level 1 ReLU)
python scripts/iterative_query.py experiment=simple_test job_name=my_test

# 完整运行
python scripts/iterative_query.py job_name=my_experiment \
    task_set.level=1 max_iters=3 inference.server_type=intel_gnai
```

**自定义 Task 运行**:
```bash
python scripts/run_custom_task.py \
    custom_task=/path/to/your/task_folder \
    job_name=my_custom_job \
    task_name=my_op
```

**单次评估（无 LLM）**:
```bash
# SYCL kernel
python scripts/run_and_check.py \
    +ref_arch_src_path=path/to/reference.py \
    +kernel_src_path=path/to/kernel.sycl

# Triton kernel (NVIDIA & Intel)
python scripts/run_and_check.py \
    +ref_arch_src_path=path/to/reference.py \
    +kernel_src_path=path/to/kernel.py language=triton
```

**生成 Baseline 时间**:
```bash
python scripts/generate_baseline_time.py --name intel_arc
```

**Web UI / Config 生成器**:
http://codegen-head1.imu.intel.com:8889/job/submission

---

## 5. 配置系统 (Hydra)

配置基于 [Hydra](https://hydra.cc/) 框架。可通过 Config 生成器创建: http://codegen-head1.imu.intel.com:8889/job/submission

模板参考: https://github.com/intel-sandbox/kernelfoundry.templates/tree/main/config_templates

```
configs/
├── run.yaml                    # 主配置入口
├── paths/default.yaml          # 路径配置
├── inference/
│   ├── server.yaml             # 单模型推理
│   ├── ensemble.yaml           # 多模型集成
│   └── local.yaml              # 本地模型
├── prompt/
│   ├── default.yaml            # Prompt 配置
│   └── meta_prompting.yaml     # 元 prompt 进化
├── task_set/default.yaml       # 任务集配置
├── database/                   # 进化数据库配置
├── experiment/                 # 实验预设配置
└── ...
```

### 5.1 必填顶层参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `task_name` | string | 操作名称（如 "relu", "gemm_int8"） |
| `job_name` | string | 本次运行名称 |
| `language` | string | 目标语言: `"SYCL"`, `"CUDA"`, `"triton"`, `"OCL"` |
| `gpu_arch` | string | 目标架构: `"ptl"`, `"lnl"`, `"bmg"`, `"Ampere"` |

> 注意: `gpu_arch` 与 `language` 耦合，如 `"Ampere"` 只能与 `"CUDA"` 配合使用。

### 5.2 重要顶层参数

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `max_iters` | int | 最大迭代次数 | 3 |
| `branches_per_iteration` | int | 每轮生成的 kernel 数量（LLM 调用次数） | 1 |
| `evolve_mode` | bool | 是否启用进化算法（否则线性改进上一轮 kernel） | false |
| `stop_once_correct` | bool | 找到正确 kernel 后停止（适合翻译任务） | false |
| `build_timeout` | int | 编译超时（秒） | 200 |
| `test_timeout` | int | 测试执行超时（秒），含正确性测试、benchmark、profiling | 300 |
| `test_reference` | bool | 是否测试 reference（计算 speedup 需要） | true |
| `has_build_step` | bool | 是否有编译步骤 | true |
| `has_reference_build_step` | bool | reference 是否需要编译（PyTorch 代码设为 false） | true |
| `use_feedback_llm` | bool | 使用第二个 LLM 分析评估日志 | false |
| `kernels_iter_0_path` | string | 设为 `"best"` 从数据库中取最佳 kernel 作为起点 | null |

### 5.3 LLM 推理配置 (`inference` 段)

默认使用模型集成（ensemble），以列表形式提供多个 server:

```yaml
inference:
  _target_: kernelgen.inference_server.LLMEnsemble
  servers:
    - _target_: kernelgen.inference_server.InferenceServer
      server_type: intel_gnai      # 服务类型
      model_name: gpt-5.2          # 模型名称
      max_tokens: 5000             # 最大生成 token 数
      temperature: 0.0             # 采样温度 (0.0=确定性)
      num_completions: 1           # 每次生成数量
      verbose: False               # 详细日志
      timeout: 400                 # GNAI 请求超时(秒)
  weights: uniform
```

> 如遇推理超时，增大 `timeout` 值。

### 5.4 Prompt 配置 (`prompt` 段)

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `reference_language` | string | Reference 的语言 | Pytorch |
| `num_optimization_tips` | int | 提示中包含的优化策略数量 | 2 |
| `include_inspirations` | bool | 是否包含灵感示例（仅 evolve_mode） | true |
| `include_best_program` | bool | 是否包含当前最佳 kernel（仅 evolve_mode） | true |
| `include_hardware_specs` | bool | 是否包含硬件规格信息 | true |
| `allow_templated` | bool | 是否允许模型写带模板参数的 kernel | true |

#### RAG 配置

RAG 以列表形式配置，支持多种数据库:

```yaml
prompt:
  rag:
    - _target_: kernelgen.database.rag_pytorch_to_sycl.RagPytorchToSycl
      top_k: 1                    # 检索示例数
      restrict_to_level: null     # 限定 KernelBench 级别 (1/2/3)
      restrict_to_correct: true   # 只包含正确的 kernel
      min_runtime_improvement: 1.1 # 最低加速比阈值
      rag_mode: closest_match     # closest_match | random
      include_prob: 1.0           # 使用 RAG 的概率
    - _target_: kernelgen.database.rag_esimd.RagESIMD
      top_k: 1
      rag_mode: closest_match     # closest_match | random | guide
      include_prob: 1.0
    - _target_: kernelgen.database.rag_joint_matrix.RagJointMatrix
      top_k: 1                    # Joint Matrix 扩展指南 + 6 个示例
      rag_mode: guide             # closest_match | random | guide
    - _target_: kernelgen.database.rag_hecbench.RagHeCBench
      top_k: 1                    # HeCBench SYCL kernel 数据库
      rag_mode: closest_match
      max_length: 10000           # 示例最大字符长度
```

| RAG 数据库 | 内容 |
|------------|------|
| `RagPytorchToSycl` | 数千个已生成的高性能 SYCL kernels |
| `RagESIMD` | ESIMD (Explicit SIMD) 示例和指南 |
| `RagJointMatrix` | Intel Joint Matrix 扩展(XMX 单元)指南 + 示例 |
| `RagHeCBench` | HeCBench benchmark 的 SYCL kernels |

### 5.5 进化算法配置 (`database.config` 段)

当 `evolve_mode: true` 时激活:

```yaml
evolve_mode: true
database:
  config:
    num_top_programs: 1           # 包含的顶级程序数
    num_diverse_programs: 0       # 包含的多样性程序数
    population_size: 1000         # 种群大小
    archive_size: 100             # 精英存档大小
    num_islands: 4                # 进化岛数量
    programs_per_island: 10       # 每岛程序数
    num_inspirations: 2           # 灵感示例数
    elite_selection_ratio: 0.1    # 精英选择比例
    exploration_ratio: 0.2        # 探索率
    exploitation_ratio: 0.7       # 利用率
    diversity_metric: edit_distance  # edit_distance | feature_based
    feature_dimensions: ["complexity", "diversity"]
    feature_bins: 10              # 每维度 bin 数
    migration_interval: 10        # 迁移间隔(代)
    migration_rate: 0.1           # 迁移比例
```

### 5.6 评估配置 (`eval_config` 段)

```yaml
eval_config:
  verbose: False
  build_timeout: 200              # 编译超时
  test_timeout: 300               # 测试超时
  num_perf_trials: 100            # 性能测量试验次数
  warmup_min_iters: 10            # 预热最小迭代
  warmup_min_time: 0.1            # 预热最小时间(秒)
  inner_loop_min_time: 0.01       # kernel 最小执行时间
```

### 5.7 实验预设

| 预设 | 用途 |
|------|------|
| `simple_test` | 快速验证：Level 1 单问题，3次迭代 |
| `translation` | 翻译模式：Feedback LLM + RAG + 10次迭代 + stop_once_correct |
| `optimize` | 进化优化：MAP-Elites + 集成模型 + RAG + 5轮×4分支 |
| `high_performance` | 全功能：Feedback + RAG + 多completion + 5轮 |
| `single_shot` | 单次基准：1次推理不迭代 |

---

## 6. 自定义 Task 开发

### 6.1 Task 目录结构

```
my_task/
├── config.yaml        # 任务配置（必需）
├── conftest.py        # pytest fixture 导入（必需）
├── task.py            # 测试类定义（必需）
├── *_kernel.EXT       # Kernel 源码（被LLM优化的部分）
└── *_reference.EXT    # Reference 实现（可选）
```

### 6.2 config.yaml 示例

```yaml
task_name: relu
job_name: test_relu_template
gpu_arch: bmg
language: SYCL
test_reference: true
has_reference_build_step: false

hyperparameters:
  buildtime: null
  runtime: null
```

### 6.3 conftest.py

```python
"""固定内容，不需修改"""
from kernelfoundry.conftest import *
```

### 6.4 task.py 结构

```python
import pytest
import torch
from pathlib import Path
from kernelfoundry.custom_test import CustomTest
from kernelfoundry.testing import assert_allclose

# [REFERENCE_START]
def reference_kernel(a):
    return torch.relu(a)
# [REFERENCE_END]

"""
[USER_INSTRUCTIONS_START]
Write a SYCL kernel for relu.
[USER_INSTRUCTIONS_END]
"""

@pytest.fixture(scope="session")
def kernel(use_reference):
    if use_reference:
        return reference_kernel
    else:
        import my_kernel
        return my_kernel.forward

class TestMyOp(CustomTest):
    def build(self, gpu_arch) -> list[str]:
        return self.compile_torch_extension(
            extension_name="my_kernel",
            src="my_kernel.sycl",
            output_dir=Path(__file__).parent,
            gpu_arch=gpu_arch,
        )

    def build_reference(self, gpu_arch) -> list[str]:
        return []

    def test_correctness(self, kernel, device):
        """正确性测试 - 必须至少一个"""
        x = torch.randn(1024, 1024)
        expected = reference_kernel(x)
        result = kernel(x.to(device))
        assert_allclose(result, expected.to(device))

    @pytest.mark.performance
    def test_benchmark(self, kernel, device, measure_runtime_torch):
        """性能测试 - 必须恰好一个带 @pytest.mark.performance"""
        x = torch.randn(1024, 1024).to(device)
        measure_runtime_torch(kernel, device, args=(x,))
```

### 6.5 测试编写规则（重要）

#### 正确性测试
- 至少需要一个正确性测试
- 测试名必须以 `test_` 开头
- 所有未标记 `@pytest.mark.performance` 的测试都视为正确性测试
- 推荐使用 `kernelfoundry.testing.assert_allclose` 比较结果

#### 性能 Benchmark 测试
- **必须且只能有一个**带 `@pytest.mark.performance` 装饰器的测试
- 推荐使用内置 fixture `measure_runtime_torch`（自动处理数据传输、预热、结果存储）
- 自定义 benchmark 必须使用 `profile_store` fixture 存储结果，使用 `profiler_session` 上下文

```python
@pytest.mark.performance
def test_benchmark(self, kernel, device, data, measure_runtime_torch):
    """标准方式 - 推荐"""
    measure_runtime_torch(kernel, device, args=data)
```

### 6.6 Kernel 文件标记

```c
// [EVOLVE_START]
// LLM 生成的代码放在这两个标记之间
// [EVOLVE_END]
```

### 6.7 本地调试

```bash
# 1. 编译 kernel (执行 build 函数)
python task.py

# 2. 测试 reference 是否正确（最重要的验证）
pytest --ref -s task.py

# 3. 测试 kernel（需要 EVOLVE 中有可编译代码）
pytest -s task.py
```

> 注意: 如果 EVOLVE block 中只有骨架/占位代码，kernel 编译可能失败，这是正常的。

### 6.8 可用 Use Cases

| 模板 | 描述 |
|------|------|
| `pytorch_to_sycl` | PyTorch → SYCL kernel |
| `sycl_to_sycl` | 优化已有 SYCL kernel |
| `cuda_to_sycl` | CUDA → SYCL 迁移 |
| `pytorch_to_ocl` | PyTorch → OpenCL kernel |
| `ocl_to_ocl` | 优化已有 OpenCL kernel |
| `model_layer_sycl` | 模型层级 SYCL 优化 |

---

## 7. 获得最佳性能的策略

根据官方 Wiki 推荐，以下策略可逐步提升 kernel 生成质量：

### 7.1 从最佳 kernel 继续

如已有之前运行生成的好 kernel，可以从最佳结果继续:
```yaml
kernels_iter_0_path: best
```

### 7.2 启用进化算法 + 更多迭代

```yaml
evolve_mode: true
branches_per_iteration: 3
max_iters: 15
```

### 7.3 使用更强大和多样的模型

使用多模型集成获得最佳效果:
```yaml
inference:
  _target_: kernelgen.inference_server.LLMEnsemble
  servers:
    - _target_: kernelgen.inference_server.InferenceServer
      server_type: intel_gnai
      model_name: claude-4-5-sonnet
      temperature: 0.3
      max_tokens: 6500
    - _target_: kernelgen.inference_server.InferenceServer
      server_type: intel_gnai
      model_name: gpt-5.2
      temperature: 0.3
      max_tokens: 6500
    - _target_: kernelgen.inference_server.InferenceServer
      server_type: intel_gnai
      model_name: gpt-5.1-codex-max
      temperature: 0.3
      max_tokens: 6500
  weights: uniform
```

### 7.4 注入 RAG 知识（实验性）

```yaml
prompt:
  rag:
    - _target_: kernelgen.database.rag_pytorch_to_sycl.RagPytorchToSycl
      top_k: 1
      restrict_to_correct: true
      min_runtime_improvement: 1.1
      rag_mode: closest_match
```

### 7.5 使用 Feedback LLM 修复错误

如观察到大量不正确的 kernel，启用 Feedback LLM 解释评估日志：
```yaml
use_feedback_llm: true
```

---

## 8. 进化优化 (MAP-Elites)

### 8.1 概念

MAP-Elites 是一种 Quality-Diversity 算法，在 KernelFoundry 中用于探索不同优化策略的 kernel 解空间。

### 8.2 行为特征维度 (3D Grid: 4×4×4 = 64 niches)

| 维度 | 描述 | 等级 (0-3) |
|------|------|-----------|
| `memory_opt` | 内存层次利用 | 无优化 → 高级缓存策略 |
| `compute_opt` | 算法效率 | 朴素 → 高度优化算法 |
| `parallelism_opt` | 并行粒度 | 基础 → 精细并行控制 |

### 8.3 配置

```yaml
evolve_mode: true
branches_per_iteration: 4
max_iters: 20

# QD Gradient Tracking
use_gradient_tracking: true
gradient_sampling_weight: 0.3

# 优化感知 Prompting
use_optimization_aware_prompting: True
exploration_strategy: mutate  # mutate/intensify/diversify
```

---

## 9. Prompt 工程

### 9.1 Prompt 构建流程

**第一轮迭代 Prompt 组成**:
1. User Instructions (`[USER_INSTRUCTIONS_START/END]`)
2. Reference Code (`[REFERENCE_START/END]`)
3. RAG Examples / Inspirations
4. Initial code / skeleton (EVOLVE block 中的初始代码)

**后续迭代 Prompt 组成**:
1. User Instructions
2. Reference Code
3. RAG Examples / Inspirations
4. 上一轮生成的 kernel + 评估日志（含 Profiler 输出）
5. 当前最佳 kernel（evolve_mode 时）
6. 来自 kernel 数据库的 parent program
7. Hardware Specs
8. Optimization Tips

### 9.2 配置选项

```yaml
prompt:
  diff_format: False          # 是否使用 diff 格式
  num_optimization_tips: 2    # 优化建议数量
  include_inspirations: True  # 包含灵感片段 (仅 evolve_mode)
  include_best_program: True  # 包含当前最佳程序 (仅 evolve_mode)
  include_hardware_specs: True
  allow_templated: True       # 允许模板化 kernel
  reference_language: Pytorch
  rag: []                     # RAG 数据库列表
```

---

## 10. MCP Server (IDE 集成)

### 10.1 安装

```bash
pip install 'kernelgen[mcp] @ git+https://github.com/intel-sandbox/kernelfoundry.internal.git'
# 或从源码
pip install ".[mcp]"
```

### 10.2 VS Code 配置

Ctrl+Shift+P → `MCP: Add Server`:
- Command: `path/to/python -m kernelgen.mcp`
- Server ID: `kernelfoundry-mcp`

`mcp.json` 环境变量:
```json
{
    "kernelfoundry-mcp": {
        "env": {
            "KERNELFOUNDRY_TOKEN": "YOUR_GNAI_TOKEN",
            "KERNELFOUNDRY_SERVER_URL": "http://isl-igpu1.rr.intel.com:8889",
            "KERNELFOUNDRY_IDSID": "YOUR_IDSID"
        }
    }
}
```

### 10.3 工具

- **`build_and_test(folder_path)`**: 打包本地任务文件夹，提交到 KernelFoundry 服务器进行验证，返回构建/测试结果。

---

## 11. Benchmark 评估 (kernel-eval)

### 11.1 任务组

| 组 | 描述 |
|----|------|
| oneDNN | SYCL kernel vs oneDNN 黄金标准 |
| oneDAL | SYCL kernel vs oneDAL 参考实现 |
| OpenVino | OCL/SYCL kernel vs PyTorch 参考 |

### 11.2 Task 结构

```
tasks/<GROUP>/<TASK_NAME>/
├── config.yaml
├── conftest.py
├── *_kernel.EXT
├── *_reference.EXT
├── task.py
├── build_kernel.sh          # 可选: AOT 编译
├── build_reference.sh       # 可选: 参考编译
├── CMakeLists_kernel.txt    # 可选
└── CMakeLists_reference.txt # 可选
```

### 11.3 性能评分标准

| Score | 含义 |
|-------|------|
| 0 | 语法错误 |
| 1 | 编译失败 |
| 2 | 编译成功但运行时错误 |
| 3 | Shape 不匹配 |
| 4 | 数值不匹配 |
| 5 | 正确性通过 |

---

## 12. KernelBench 任务层级

| Level | 描述 | 问题数 |
|-------|------|--------|
| Level 1 🧱 | 单 kernel 算子 (Conv, MatMul, LayerNorm) | 100 |
| Level 2 🔗 | 简单融合模式 (Conv+Bias+ReLU) | 100 |
| Level 3 ⚛️ | 完整模型架构 (MobileNet, VGG, MiniGPT) | 50 |
| Level 4 🤗 | HuggingFace 完整模型 | - |

---

## 13. 常用配置模板 (templates)

### 13.1 简单测试

```yaml
# config_templates/simple_test.yaml
task_name: relu
job_name: testing
language: SYCL
gpu_arch: lnl
inference:
  servers:
  - _target_: kernelgen.inference_server.InferenceServer
    server_type: intel_gnai
    model_name: gpt-5-mini
    temperature: 0.0
    max_tokens: 6500
evolve_mode: false
max_iters: 3
```

### 13.2 翻译模式

```yaml
# config_templates/translation.yaml
inference:
  servers:
  - server_type: intel_gnai
    model_name: gpt-5.2
    temperature: 0.0
use_feedback_llm: true
evolve_mode: false
max_iters: 10
stop_once_correct: true
prompt:
  rag:
    - _target_: kernelgen.database.rag_pytorch_to_sycl.RagPytorchToSycl
      top_k: 1
      restrict_to_correct: True
```

### 13.3 进化优化

```yaml
# config_templates/optimize_evolve.yaml
evolve_mode: true
branches_per_iteration: 4
max_iters: 20
store_generated_kernels_in_db: true
inference:
  _target_: kernelgen.inference_server.LLMEnsemble
  servers:
    - model_name: claude-4-5-sonnet
    - model_name: gpt-4.1
    - model_name: gpt-5.2
  weights: uniform
use_gradient_tracking: true
use_optimization_aware_prompting: True
exploration_strategy: mutate
```

---

## 14. OpenCL Task 示例

适用于 `ocl_to_ocl` 或 `pytorch_to_ocl` 场景：

```yaml
# config.yaml
task_name: gemm_ocl
job_name: test_gemm_ocl
gpu_arch: ptl
language: OCL
prompt:
  reference_language: OCL
test_reference: true
evolve_mode: false
max_iters: 3
```

```python
# task.py 关键结构
import pyopencl as cl
from kernelfoundry.custom_test import CustomTest

@pytest.fixture(scope="session")
def kernel(use_reference, ocl_queue):
    filename = "gemm_reference.cl" if use_reference else "gemm_kernel.cl"
    return initialize_gemm_kernel(filename, ocl_queue)

class TestGemmOCL(CustomTest):
    def build(self, gpu_arch) -> list[str]: ...
    def test_correctness(self, kernel, ocl_queue): ...
    @pytest.mark.performance
    def test_benchmark(self, kernel, ocl_queue, measure_runtime_ocl): ...
```

---

## 15. Fine-tuning & GRPO

位于 `kernelgen/finetuning/` 目录，支持：
- 使用生成数据进行 SFT (Supervised Fine-Tuning)
- GRPO (Group Relative Policy Optimization) 强化学习
- 自定义 reward 函数（编译成功、正确性、运行时改进）

相关配置: `configs/finetune/`, `configs/grpo_exp/`

---

## 16. 数据库

### 16.1 表结构 (SQLAlchemy)

核心表: `Kernel`, `Task`, `Job`

关键字段:
- `input_code`: 参考代码
- `output_code`: 生成的 kernel
- `input_language` / `output_language`
- `gpu_arch`: 目标架构
- `compiled` / `correctness` / `runtime` / `runtime_improvement`

### 16.2 存储数据库结果

通过配置 `store_generated_kernels_in_db: true` 在进化运行中自动存储。

---

## 17. 调试与开发

### 17.1 验证单个 kernel

```bash
python -m pytest task.py                  # 测试 LLM 生成的 kernel
python -m pytest task.py --use-reference  # 测试 reference
python -m pytest task.py -k "test_correctness"  # 只跑正确性
python -m pytest task.py -k "performance"       # 只跑性能
```

### 17.2 查看可用模型

```bash
python scripts/check_gnai_models.py
```

### 17.3 日志

运行日志保存在 `runs/<job_name>/` 目录下，包含:
- `controller.log`: 主控制器日志
- 各迭代的 kernel 代码和评估结果

### 17.4 kernelfoundry Python 包安装

```bash
# 从 PyPI (内部 registry)
pip install kernelfoundry --extra-index-url https://isl-igpu1.rr.intel.com/whl --trusted-host isl-igpu1.rr.intel.com

# 从源码
cd kernelfoundry.kernel-eval && pip install .
```

---

## 18. 关键概念术语

| 术语 | 含义 |
|------|------|
| EVOLVE block | Kernel 文件中由 `[EVOLVE_START/END]` 标记的可修改区域 |
| REFERENCE block | `[REFERENCE_START/END]` 标记的参考实现 |
| USER_INSTRUCTIONS | `[USER_INSTRUCTIONS_START/END]` 标记的自定义指令 |
| Program | 一个 kernel 代码 + 评估结果的组合 |
| EvalResult | 编译/正确性/运行时的评估结果 |
| CustomTask | 包含所有 task 文件和元数据的对象 |
| CustomTest | pytest 测试基类，task.py 中的测试类需继承此类 |
| Feedback LLM | 分析评估日志生成摘要的辅助 LLM |
| MAP-Elites | 基于行为特征网格的 Quality-Diversity 进化算法 |
| QD Gradient | 基于转移历史估计搜索方向的梯度追踪 |

---

## 19. 典型开发工作流

1. **创建 Task**: 参照模板 (`kernelfoundry.templates/task_templates/`) 创建任务目录
2. **本地验证**: `python -m pytest task.py --use-reference` 确认 reference 可运行
3. **编写 config.yaml**: 配置目标语言、架构、推理模型
4. **运行 Pipeline**:
   - 简单翻译: `python scripts/run_custom_task.py custom_task=./my_task`
   - 进化优化: 设置 `evolve_mode: true`
5. **分析结果**: 查看 `runs/<job_name>/` 下的日志和生成的 kernels
6. **MCP 集成**: 通过 VS Code MCP server 直接提交任务

---

## 参考资源

- [Intel Wiki](https://wiki.ith.intel.com/spaces/VCL/pages/4501743035/Kernel+Foundry) (需内部访问)
- [KernelBench 论文](https://arxiv.org/abs/2502.10517)
- GitHub: `intel-sandbox/kernelfoundry.internal`
- GitHub: `intel-sandbox/kernelfoundry.kernel-eval`
- GitHub: `intel-sandbox/kernelfoundry.templates`
