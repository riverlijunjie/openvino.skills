import pytest
import numpy as np
from typing import Any
import pyopencl as cl
from pathlib import Path
from kernelfoundry.custom_test import CustomTest  # Tests must derive from CustomTask

# Fixtures provided via conftest.py from kernelfoundry: use_reference, ocl_queue, profile_store, request


@pytest.fixture(scope="session")
def ocl_queue():
    """Fixture to create an OpenCL command queue for GPU execution."""
    # Find the first GPU device across all platforms
    device = None
    for platform in cl.get_platforms():
        devs = platform.get_devices(cl.device_type.GPU)
        if devs:
            device = devs[0]
            break
    assert device is not None, "No OpenCL GPU device found"
    ctx = cl.Context([device])
    queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
    return queue


def get_data_on_device(inp_shape: tuple, ocl_queue_arg: cl.CommandQueue) -> tuple[tuple[Any, ...], np.ndarray]:
    """
    Creates random test data and moves it to device

    Args:
        inp_shape (tuple): Shape of the input matrices (m, k, n)
        ocl_queue_arg (cl.CommandQueue): OpenCL command queue for device transfers

    Returns:
        tuple[tuple[Any, ...], np.ndarray]: A tuple containing the input arguments (on device if not using reference) and the expected result as a numpy array
    """
    # create numpy arrays on host using half precision inputs
    m, k, n = inp_shape
    a = (np.random.randn(m, k) - 0.5).astype(np.float16)
    b = (np.random.randn(k, n) - 0.5).astype(np.float16)
    # for testing with respect to standard matmul (accumulate in float32):
    expected = np.matmul(a.astype(np.float32), b.astype(np.float32))

    # create device buffers and upload data
    a_flat = a.flatten()
    b_flat = b.flatten()
    a_ocl = cl.Buffer(ocl_queue_arg.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=a_flat)
    b_ocl = cl.Buffer(ocl_queue_arg.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=b_flat)
    out_ocl = cl.Buffer(
        ocl_queue_arg.context,
        cl.mem_flags.WRITE_ONLY,
        size=m * n * np.dtype(np.float16).itemsize,
    )
    return (a_ocl, b_ocl, out_ocl, np.int32(m), np.int32(k), np.int32(n)), expected


def initialize_gemm_kernel(filename: str, queue: cl.CommandQueue):
    """Helper function to initialize a GEMM OCL kernel (either from kernel fixture or for correctness test)"""
    import math
    # load OCL kernel from text
    _CL_SRC = Path(__file__).parent / filename
    src_text = _CL_SRC.read_text()
    program = cl.Program(queue.context, src_text).build()
    knl = program.gemm
    if filename == "gemm_kernel.cl" and "reqd_work_group_size" in src_text:
        # DPAS kernel: sub-group size 16, each sub-group computes 32x32 tile (N-register-blocked)
        def run(a, b, c, m, k, n):
            m_int, n_int = int(m), int(n)
            global_size = (int(math.ceil(n_int / 32)) * 16, int(math.ceil(m_int / 32)))
            local_size = (16, 1)
            return knl(queue, global_size, local_size, a, b, c, np.int32(m), np.int32(k), np.int32(n))
        return run
    else:
        # Plain GEMM kernel/reference kernel: one work-item per output element
        return lambda a, b, c, m, k, n: knl(queue, (int(n), int(m)), None, a, b, c, np.int32(m), np.int32(k), np.int32(n))


@pytest.fixture(scope="session")
def kernel(use_reference, ocl_queue):
    """Returns a callable kernel(a, b) -> np.ndarray that runs on the GPU."""
    if use_reference:
        filename = "gemm_reference.cl"
    else:
        filename = "gemm_kernel.cl"
    return initialize_gemm_kernel(filename, ocl_queue)


class TestGemmOCL(CustomTest):
    """A custom task for gemm kernels using pyopencl."""

    def build(self, gpu_arch) -> list[str]:
        """Simply call initialize_gemm_kernel to validate compilation of the kernel at build time."""
        # Create build-time context for compilation validation (can be cpu)
        device = cl.get_platforms()[0].get_devices()[0]
        ctx = cl.Context([device])
        build_queue = cl.CommandQueue(ctx)

        initialize_gemm_kernel("gemm_kernel.cl", build_queue)

    @pytest.mark.parametrize("_run", range(3))
    def test_correctness_wrt_pytorch(self, kernel, ocl_queue, _run):
        # generate random data
        args, expected = get_data_on_device((128, 256, 512), ocl_queue)

        # run kernel
        kernel(*args)

        # move result back to host
        _, _, out_buf, m, _, n = args
        result_flat = np.empty(m * n, dtype=np.float16)
        cl.enqueue_copy(ocl_queue, result_flat, out_buf)
        result = result_flat.reshape((m, n)).astype(np.float32)

        # check shape
        assert result.shape == expected.shape, f"Output shape {result.shape} != expected shape {expected.shape}"
        # check values (rtol=2e-2 to accommodate fp16 DPAS precision over K=256 accumulations)
        assert np.allclose(result, expected, rtol=2e-2, atol=2e-2), "Matrix multiply result is incorrect."

    @pytest.mark.parametrize("_run", range(3))
    def test_correctness_wrt_reference(self, kernel, ocl_queue, _run):
        # generate random data
        args, _ = get_data_on_device((128, 256, 512), ocl_queue)

        # run kernel
        kernel(*args)

        # move result back to host
        _, _, out_buf, m, _, n = args
        result_flat = np.empty(m * n, dtype=np.float16)
        cl.enqueue_copy(ocl_queue, result_flat, out_buf)
        result = result_flat.reshape((m, n)).astype(np.float32)

        # inititalize reference
        reference_kernel_function = initialize_gemm_kernel("gemm_reference.cl", ocl_queue)
        # run reference
        reference_kernel_function(*args)

        # move reference result back to host
        reference_result_flat = np.empty(m * n, dtype=np.float16)
        cl.enqueue_copy(ocl_queue, reference_result_flat, out_buf)
        expected = reference_result_flat.reshape((m, n)).astype(np.float32)

        # check values (rtol=2e-2 to accommodate fp16 DPAS precision over K=256 accumulations)
        assert np.allclose(result, expected, rtol=2e-2, atol=2e-2), "Matrix multiply result is incorrect."

    @pytest.mark.performance
    @pytest.mark.parametrize("m, k, n", [(128, 256, 512), (256, 512, 1024), (1024, 2048, 2560)])
    def test_benchmark(self, kernel, ocl_queue, measure_runtime, m, k, n):
        """Benchmark: times only the GPU kernel dispatch — no host<->device copies."""
        args, _ = get_data_on_device((m, k, n), ocl_queue)
        measure_runtime(
            kernel,
            args=args,
            sync_fn=ocl_queue.finish,
            auto_replicate_inputs_size=0,
        )


if __name__ == "__main__":
    """Optional build when running this file directly during development."""
    task = TestGemmOCL()
    task.build(CustomTest.get_machine_gpu_arch())
