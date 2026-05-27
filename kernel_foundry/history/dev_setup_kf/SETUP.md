# KernelFoundry Local Server Setup Guide

本文档总结了在本地机器上搭建 KernelFoundry 服务的步骤和注意事项，基于以下三个仓库：
- `kernelfoundry.internal` — 主代码生成与优化 pipeline
- `kernelfoundry.kernel-eval` — 内核评估/benchmark 工具
- `kernelfoundry.templates` — 任务模板

---

## 1. 系统要求

| 项目 | 要求 |
|------|------|
| OS | Linux (x86_64) |
| Python | 3.12.9 (推荐) |
| Conda | Miniforge / Mambaforge / Anaconda |
| GPU | Intel (Arc/dGPU) 或 NVIDIA (CUDA) |
| 编译器 | gcc/g++, cmake, ninja |

---

## 2. Intel 硬件设置

### 2.1 安装/验证显卡驱动和 oneAPI

```bash
cd kernelfoundry.internal
./scripts/setup/setup_oneapi.sh --install --oneapi 2025.1 --device A770
# 或仅验证：
./scripts/setup/setup_oneapi.sh --verify --oneapi 2025.1 --device A770
```

### 2.2 经过验证的组合

| oneAPI 版本 | PyTorch 版本 | 状态 |
|------------|-------------|------|
| 2025.0 | 2.7.0 | ✅ |
| 2025.1 | 2.7.1 | ❌ |
| **2025.1** | **2.8.0** | ✅ (默认) |
| 2025.2 | 2.8.0 | ❌ |

> **注意**: README 中列出的最新 PyTorch 版本为 2.9.0 (requirements_intel.txt)，请根据实际情况选择经过验证的组合。

### 2.3 设置 Python 环境

```bash
./scripts/setup/setup_repo.sh --platform intel --machine local
```

这将创建名为 `kernel-intel-py312` 的 conda 环境并安装所有依赖。

### 2.4 Source oneAPI 环境变量

```bash
source /opt/intel/oneapi/2025.0/oneapi-vars.sh
```

> 每次打开新终端时都需要 source 此文件。

### 2.5 设置 unitrace (性能分析)

```bash
cd ..  # 在 kernelfoundry.internal 旁边 clone
git clone https://github.com/intel/pti-gpu.git
pushd pti-gpu/tools/unitrace
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_WITH_MPI=OFF ..
make

# 创建 symlink 到 conda bin
unitrace_path=$PWD/unitrace
cd $CONDA_PREFIX/bin
ln -s $unitrace_path
popd

# 允许非 root 用户 profiling（每次重启后需重新执行）
sudo sh -c 'echo 0 > /proc/sys/dev/i915/perf_stream_paranoid'
sudo sh -c 'echo 0 > /proc/sys/dev/xe/observation_paranoid'
```

---

## 3. NVIDIA 硬件设置

```bash
cd kernelfoundry.internal
./scripts/setup/setup_repo.sh --platform cuda --machine local
```

这将创建名为 `kernel-cuda-py312` 的 conda 环境。PyTorch 通过 `--extra-index-url https://download.pytorch.org/whl/cu129` 安装。

---

## 4. 手动安装（不使用 setup_repo.sh）

如果不使用自动脚本，可手动执行以下步骤：

```bash
# 创建 conda 环境
conda create --name kernel_intel python=3.12.9  # 或 kernel_cuda
conda activate kernel_intel

# 安装 libclang (代码解析必需)
conda install libclang=20

# 安装 PyTorch
# Intel:
pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/xpu
# NVIDIA:
pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu129

# 安装主仓库依赖
cd kernelfoundry.internal
pip install -r requirements.txt
pip install -e .
```

---

## 5. 安装 kernelfoundry.kernel-eval

```bash
cd kernelfoundry.kernel-eval
pip install .
```

依赖: `numpy>=1.26`, `pytest`, `pytest-dependency`, `ninja`

如需运行 PyTorch 相关任务，确保已安装对应平台的 torch：
```bash
# Intel:
pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/xpu
# NVIDIA:
pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu129
```

---

## 6. 安装 kernelfoundry.templates

templates 仓库本身不需要安装，但运行模板任务需要 `kernelfoundry` 包：

```bash
# 方式一：从内部 PyPI 安装
pip install kernelfoundry --extra-index-url https://isl-igpu1.rr.intel.com/whl --trusted-host isl-igpu1.rr.intel.com

# 方式二：从源码安装（kernelfoundry.kernel-eval 仓库）
cd kernelfoundry.kernel-eval
pip install .
```

---

## 7. LLM API 配置

设置 API 密钥环境变量：

```bash
# GNAI (Intel):
export GNAI_TOKEN="your_gnai_token"

# Denvr:
export BASE_URL="https://api.inference.denvrdata.com/v1/"
export OPENAI_API_KEY="denvr_api_key_here"

# IBM:
export IBM_API_TOKEN="ibm_token_here"
```

获取 GNAI token：访问 `https://gfx-assets.intel.com/auth/oauth2/sso`，复制返回的 `access_token`。
或使用脚本 `scripts/get_gnai_token.py` (需要 authlib 包)。

---

## 8. 推荐的 .env 文件

在 `kernelfoundry.internal` 根目录创建 `.env` 文件（**不要提交到 git**）：

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

> `import autoroot` 语句会自动加载项目根目录的 `.env` 文件中的环境变量。

---

## 9. 运行实际任务

KernelFoundry 支持两种运行方式：**KernelBench 内置任务** 和 **自定义任务 (Custom Task)**。

### 9.1 运行 KernelBench 内置任务

KernelBench 任务按难度分 4 级（Level 1-4），使用 `iterative_query.py` 或 `run_custom_task.py` 运行：

```bash
cd kernelfoundry.internal

# 快速验证（Level 1 ReLU，3 轮迭代）
python scripts/iterative_query.py experiment=simple_test job_name=my_test

# 指定 level 和模型
python scripts/iterative_query.py \
  job_name=test_l1 \
  task_set.level=1 \
  max_iters=3 \
  inference.server_type=intel_gnai
```

### 9.2 运行自定义任务 (Custom Task)

自定义任务是独立的目录，包含 `config.yaml`、`task.py`、`conftest.py` 和内核源文件。

**步骤 1: 准备任务目录**

以 `kernelfoundry.templates/task_templates/pytorch_to_sycl/relu` 为例：

```
my_task/
├── config.yaml          # 任务配置（任务名、语言、架构等）
├── conftest.py          # pytest fixture 配置（通常不修改）
├── relu_kernel.sycl     # 内核文件（含 EVOLVE 标记，LLM 填充此区域）
└── task.py              # 测试逻辑（参考实现、正确性验证、benchmark）
```

**步骤 2: 编写 config.yaml**

```yaml
task_name: relu                    # 任务名称
job_name: test_relu_template       # 运行名称

language: SYCL                     # 目标语言: SYCL / CUDA / OpenCL / triton
gpu_arch: bmg                      # 目标架构: bmg / lnl / dg2 / ampere 等

test_reference: true               # 是否测试参考实现
has_reference_build_step: false    # 参考实现是否需要编译

# LLM 推理配置（可选，覆盖默认）
inference:
  servers:
  - _target_: kernelgen.inference_server.InferenceServer
    server_type: intel_gnai
    model_name: gpt-5-mini
    temperature: 0.0
    max_tokens: 6500

max_iters: 3                       # 迭代轮数
evolve_mode: false                 # 是否使用进化算法
```

**步骤 3: 编写 task.py**

```python
import torch
from kernelfoundry.custom_test import CustomTest

# [REFERENCE_START]
def reference_kernel(a):
    return torch.relu(a)
# [REFERENCE_END]

"""
[USER_INSTRUCTIONS_START]
Write a SYCL kernel for relu.
[USER_INSTRUCTIONS_END]
"""

class TestRelu(CustomTest):
    def build(self, gpu_arch) -> list[str]:
        artifacts = self.compile_torch_extension(
            extension_name="relu_kernel",
            src="relu_kernel.sycl",
            output_dir=Path(__file__).parent,
            gpu_arch=gpu_arch,
        )
        return artifacts

    def test_correctness(self, kernel, device):
        x = torch.randn(1024, 1024).to(device)
        expected = reference_kernel(x.cpu()).to(device)
        result = kernel(x)
        assert torch.allclose(result, expected, rtol=1e-4, atol=1e-4)

    @pytest.mark.performance
    def test_benchmark(self, kernel, device, measure_runtime_torch):
        x = torch.randn(1024, 1024).to(device)
        measure_runtime_torch(kernel, device, args=(x,))
```

**步骤 4: 内核文件中使用 EVOLVE 标记**

```c
// relu_kernel.sycl
// [EVOLVE_START]
// LLM 将在此区域内生成代码
// [EVOLVE_END]
```

**步骤 5: 运行自定义任务**

```bash
python scripts/run_custom_task.py \
  custom_task=path/to/my_task \
  task_origin=custom \
  job_name=my_relu_test
```

### 9.3 使用 kernel-eval 运行 benchmark 任务

```bash
cd kernelfoundry.kernel-eval

# 运行特定任务组的评估
bash run_eval.sh
```

任务组包括: `onednn`（SYCL vs oneDNN）、`onedal`（SYCL vs oneDAL）、`ov_ocl`（OpenCL/SYCL vs PyTorch）。

### 9.4 运行进化优化模式 (Evolve Mode)

对于复杂内核优化，使用多分支进化策略：

```bash
python scripts/run_custom_task.py \
  custom_task=path/to/my_task \
  task_origin=custom \
  job_name=my_evolve_run \
  evolve_mode=true \
  branches_per_iteration=4 \
  max_iters=20
```

或使用配置模板：
```bash
# 将 config_templates/optimize_evolve.yaml 中的参数合并到 config.yaml
```

### 9.5 单内核手动评估（不使用 LLM）

```bash
# SYCL 向量加法
python scripts/run_and_check.py \
  +ref_arch_src_path=kernelgen/prompts/kernel_examples/pytorch_functional_ex_add.py \
  +kernel_src_path=kernelgen/prompts/kernel_examples/sycl_example_add_raw.sycl

# Triton（同时支持 NVIDIA 和 Intel）
python scripts/run_and_check.py \
  +ref_arch_src_path=kernelgen/prompts/kernel_examples/pytorch_functional_ex_add.py \
  +kernel_src_path=kernelgen/prompts/kernel_examples/triton_functional.py \
  language=triton
```

### 9.6 生成 baseline 时间

```bash
python scripts/generate_baseline_time.py --name <your_name>
# 结果保存在 results/timing/your_name
```

---

## 10. 关键概念说明

| 概念 | 说明 |
|------|------|
| `REFERENCE` 块 | `[REFERENCE_START]...[REFERENCE_END]` 标记的参考实现，用于验证正确性和性能对比 |
| `EVOLVE` 块 | `[EVOLVE_START]...[EVOLVE_END]` 标记的区域，LLM 在此生成/优化代码 |
| `USER_INSTRUCTIONS` | `[USER_INSTRUCTIONS_START]...[USER_INSTRUCTIONS_END]` 标记的用户指令 |
| `CustomTest` | 继承此类编写任务，提供 `build()`、`test_*()` 方法 |
| `evolve_mode` | 进化优化模式，多分支并行生成，逐代保留最优 |
| `task_origin` | 任务来源标识: `KernelBench` / `robust_kbench` / `custom` |
| `gpu_arch` | 目标 GPU 架构: `dg2` (Arc A770), `bmg` (Battlemage), `lnl` (Lunar Lake), `ampere` |

---

## 11. 注意事项

1. **oneAPI 版本兼容性**: 务必使用经过验证的 oneAPI + PyTorch 组合，避免不兼容问题。
2. **每次重启后**: 需重新 source oneAPI 环境变量，并重新设置 `perf_stream_paranoid` / `observation_paranoid`。
3. **libclang=20 必须通过 conda 安装**: 用于代码解析和操作，pip 安装可能版本不匹配。
4. **.env 文件安全**: 包含密码和 token，不要提交到版本控制。
5. **GPU 必需**: 运行和 profiling 内核需要 GPU；纯 LLM 推理部分可以不需要。
6. **网络代理**: 如需代理，参考 `docs/apt_proxy.md`。
7. **kernel-eval 包名**: 安装后 Python 包名为 `kernelfoundry`，不是 `kernel-eval`。
8. **任务模板使用**: templates 仓库提供多种任务类型（pytorch_to_sycl, sycl_to_sycl, pytorch_to_ocl, ocl_to_ocl, model_layer_sycl），选择合适的模板开始。
9. **数据库和消息队列**: 完整 pipeline 需要 PostgreSQL 和 RabbitMQ 服务（通过 .env 配置连接信息）。
10. **子模块**: `setup_repo.sh` 会自动初始化 git 子模块（如 LLaMA-Factory），手动安装时需注意。
