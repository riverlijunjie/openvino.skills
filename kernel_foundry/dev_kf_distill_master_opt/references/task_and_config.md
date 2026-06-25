# Authoring a Task & Configuring a Run

Distilled from `kernelfoundry.templates/` (task_templates, config_templates) and
`kernelfoundry/configs/`. This is the *user-facing* contract: what files you write to define a
kernel-optimization problem, and how you configure the search over it.

---

## 1. A task directory

The minimum (PyTorch→SYCL/CUDA style):

```
my_task/
├── config.yaml          # task + run metadata
├── conftest.py          # one line: from kernelfoundry.conftest import *   (do not edit)
├── task.py              # REFERENCE block + pytest correctness/perf tests + build()
└── my_kernel.EXT        # the EVOLVE block lives here (.sycl/.cu/.cl/.py)
```

Ahead-of-time-compiled tasks (SYCL→SYCL optimizing an existing library kernel) add
`kernel.cpp` + `reference.cpp`, `CMakeLists_kernel.txt` / `CMakeLists_reference.txt`, and
`build_kernel.sh` / `build_reference.sh`.

### `config.yaml`
```yaml
task_name: relu                 # name of the kernel task
job_name: test_relu             # name of this run/experiment
gpu_arch: bmg                   # target arch: bmg/lnl/ptl/dg2 (Intel) | Hopper/Ada/Ampere (NVIDIA)
language: SYCL                  # SYCL | CUDA | triton | OCL
test_reference: true            # benchmark the reference too → enables real speedup
has_reference_build_step: false # false when reference is plain PyTorch
hyperparameters:
  buildtime: null               # passed to build() in task.py
  runtime: null                 # exposed to tests via the `hyperparams` fixture
```

### `task.py` (the heart of the task)
```python
import pytest, torch
from pathlib import Path
from kernelfoundry import TestBase            # tests must derive from TestBase

# [REFERENCE_START]
def reference_kernel(a):
    return torch.relu(a)                      # ground truth: correctness + speedup baseline
# [REFERENCE_END]

# [USER_INSTRUCTIONS_START]
# Write a fast SYCL kernel for relu.
# [USER_INSTRUCTIONS_END]

@pytest.fixture(scope="session")
def device(): return torch.device("xpu")      # "cuda" on NVIDIA

@pytest.fixture(scope="session")
def kernel(use_reference):                    # --ref flag flips this to the reference
    if use_reference: return reference_kernel
    import relu_kernel; return relu_kernel.forward

class TestRelu(TestBase):
    def build(self, gpu_arch) -> list[str]:   # compile the EVOLVE kernel to a torch extension
        return self.compile_torch_extension(
            extension_name="relu_kernel", src="relu_kernel.sycl",
            output_dir=Path(__file__).parent, gpu_arch=gpu_arch)
    def build_reference(self, gpu_arch): return []   # PyTorch ref needs no build

    def test_correctness_values(self, kernel, device):       # correctness test (no marker)
        x = (torch.randn(1024,1024)-0.5).to(device)
        assert torch.allclose(kernel(x), reference_kernel(x), rtol=1e-4, atol=1e-4)

    @pytest.mark.performance                                  # performance test (marker!)
    def test_benchmark(self, kernel, device, measure_runtime_torch):
        x = (torch.randn(1024,1024)-0.5).to(device)
        measure_runtime_torch(kernel, device, args=(x,))     # fixture does warmup + timing
```

Rules that the harness enforces / relies on:
- Exactly one `EVOLVE` block; ≥1 correctness test; ≥1 `@pytest.mark.performance` test.
- Correctness tests assert against `reference_kernel`; perf tests call a `measure_runtime_*` fixture.
- The kernel file (`relu_kernel.sycl`) contains only the markers initially:
  `// [EVOLVE_START]` / `// [EVOLVE_END]` — the model fills the body.

You can sanity-check a task by hand exactly like the repo does:
`pytest --ref -s my_task/task.py` (reference) then `pytest -s my_task/task.py` (current kernel).

---

## 2. Run configuration (Hydra)

The pipeline is `run_custom_task.py` driven by `configs/run.yaml`, layered with
`paths / inference / database / prompt / task_set / experiment` groups. Override on the CLI:

```bash
python scripts/run_custom_task.py custom_task=/abs/path/to/my_task \
  task_origin=local_relu job_name=relu_v1 task_name=relu \
  gpu_arch=bmg language=SYCL \
  evolve_mode=true branches_per_iteration=4 max_iters=20 \
  use_optimization_aware_prompting=true \
  test_reference=true store_generated_kernels_in_db=false
```

### Three canonical run profiles (`config_templates/`)
- **translation.yaml** — *get it correct*: `evolve_mode:false`, `max_iters:10`,
  `branches_per_iteration:1`, `stop_once_correct:true`, one strong low-temp model,
  feedback-LLM on, RAG of similar correct translations.
- **optimize_evolve.yaml** — *make it fast*: `evolve_mode:true`, `branches_per_iteration:4`,
  `max_iters:20`, model **ensemble** at `temperature:0.3` (diversity), gradient tracking on,
  `use_optimization_aware_prompting:true`, `exploration_strategy:mutate`.
- **simple_test.yaml** — smoke test on a fixed task subset, `start_from_best:true`.

### Inference: single vs ensemble
```yaml
inference:                                  # one strong model (translation)
  servers:
  - {_target_: ...InferenceServer, server_type: ..., model_name: <strong>, temperature: 0.0, max_tokens: 6500}

inference:                                  # ensemble (evolution → diversity)
  _target_: ...LLMEnsemble
  servers: [ <model A>, <model B>, <model C> ]   # all at temperature 0.3
  weights: uniform
```

### Database / evolution config (`database/evolve_db_optimization_aware.yaml`)
```yaml
config:
  population_size: 1000
  num_islands: 4
  programs_per_island: 10
  num_inspirations: 2
  num_top_programs: 1
  exploration_ratio: 0.2      # sample from current island
  exploitation_ratio: 0.7     # sample from elite archive  (remaining 0.1 random)
  feature_dimensions: [memory_opt, compute_opt, parallelism_opt]   # + esimd_opt optional
  feature_bins: {memory_opt: 4, compute_opt: 4, parallelism_opt: 4}
  migration_interval: 10
  migration_rate: 0.1
  random_seed: 42
```

### Gradient config (`run.yaml`)
```yaml
use_gradient_tracking: true
gradient_sampling_weight: 0.3
gradient_config: {fitness_weight: 0.4, improvement_rate_weight: 0.4, exploration_weight: 0.2,
                  max_history: 10000, max_cell_cache: 256, checkpoint_interval: 100}
```

### Prompt config (`prompt/default.yaml`, `prompt/meta_prompting.yaml`)
```yaml
num_optimization_tips: 2
include_inspirations: true
include_best_program: true
include_hardware_specs: true
reference_language: Pytorch
meta_prompting: {enabled: false, mode: holistic, evolution_interval: 10,
                 min_samples_for_evolution: 5, exploration_rate: 0.2, max_prompts: 50}
rag: []                       # add RAG providers here for few-shot examples
```

---

## 3. Picking settings (rules of thumb)
- **Correctness first / port a kernel** → translation profile, `stop_once_correct:true`.
- **Squeeze performance from a correct kernel** → evolve profile, `start_from_best:true`,
  optimization-aware prompting + gradient on, ensemble at temp 0.3, `max_iters` 20–40.
- **More compute budget** → raise `branches_per_iteration` (wider) and/or `max_iters` (deeper);
  more `num_islands` keeps diversity at high iteration counts.
- **No reference to compare** → `test_reference:false`; fitness falls back to `1/runtime`.
- **Profiling is expensive** → `eval_config.profile_custom_model:false` to skip it for quick runs.
