"""
task.py -- defines ONE kernel-optimization problem for the KernelFoundry pipeline.

Contract enforced by the harness (task.validate()):
  * the REFERENCE block below = ground truth for correctness AND the speedup baseline
  * >= 1 correctness test (no marker) that asserts kernel(x) ≈ reference_kernel(x)
  * >= 1 performance test marked @pytest.mark.performance that calls a measure_runtime_* fixture
  * the kernel source file holds exactly one [EVOLVE_START]/[EVOLVE_END] region (the model fills it)

Hand-check before launching a run:
    pytest --ref -s task.py      # exercises the reference
    pytest      -s task.py       # exercises the current EVOLVE kernel
"""
import pytest
import torch
from pathlib import Path
from kernelfoundry import TestBase  # all task test classes derive from TestBase


# [REFERENCE_START]
def reference_kernel(a):
    # TODO: the operation to optimize. Ground truth + baseline. Keep it simple & correct.
    return torch.relu(a)
# [REFERENCE_END]


# [USER_INSTRUCTIONS_START]
# TODO (optional): free-text guidance handed verbatim to the LLM, e.g.
# "Write a fused, vectorized SYCL kernel; tile into shared local memory."
# [USER_INSTRUCTIONS_END]


@pytest.fixture(scope="session")
def device():
    return torch.device("xpu")        # "cuda" on NVIDIA


@pytest.fixture(scope="session")
def kernel(use_reference):
    # use_reference is True when pytest is invoked with --ref
    if use_reference:
        return reference_kernel
    import my_op_kernel                # TODO: must match extension_name / src below
    return my_op_kernel.forward        # TODO: the entry point your kernel exposes


def get_data(shape, device):
    x = torch.randn(shape) - 0.5
    return x.to(device), reference_kernel(x).to(device)


class TestMyOp(TestBase):
    """One custom task. Rename to match your op."""

    def build(self, gpu_arch) -> list[str]:
        # Compile the EVOLVE kernel into a torch extension (SYCL/CUDA picked by gpu_arch).
        return self.compile_torch_extension(
            extension_name="my_op_kernel",       # TODO
            src="my_op_kernel.sycl",             # TODO: the file containing the EVOLVE block
            output_dir=Path(__file__).parent,
            gpu_arch=gpu_arch,
        )

    def build_reference(self, gpu_arch) -> list[str]:
        return []                                # PyTorch reference needs no build

    # ---- correctness (no marker) -------------------------------------------------
    def test_correctness_shape(self, kernel, device):
        x, y = get_data((1024, 1024), device)
        assert kernel(x).shape == y.shape

    @pytest.mark.parametrize("_run", range(3))
    def test_correctness_values(self, kernel, device, _run):
        x, y = get_data((1024, 1024), device)
        assert torch.allclose(kernel(x), y, rtol=1e-4, atol=1e-4), "output mismatch"

    # ---- performance (MUST carry the marker) ------------------------------------
    @pytest.mark.performance
    def test_benchmark(self, kernel, device, measure_runtime_torch):
        x, _ = get_data((1024, 1024), device)
        measure_runtime_torch(kernel, device, args=(x,))   # fixture handles warmup + timing


if __name__ == "__main__":
    # Optional: build locally during development.
    t = TestMyOp()
    t.build_reference(TestBase.get_machine_gpu_arch())
    t.build(TestBase.get_machine_gpu_arch())
