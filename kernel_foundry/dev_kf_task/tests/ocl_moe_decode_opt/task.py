import pytest
import numpy as np
import pyopencl as cl
from pathlib import Path
from kernelfoundry.custom_test import CustomTest  # Tests must derive from CustomTask

# Fixtures provided via conftest.py from kernelfoundry: use_reference, ocl_queue, profile_store, request

# ===================== Model parameters (must match kernel macros) =====================
MAX_TOPK = 8
EXPERT_NUM = 128
HIDDEN_SIZE = 2048
INTERMEDIATE_SIZE = 768
GROUP_SIZE = 128          # GATE_UP_GROUP_SIZE
NUM_GROUPS = HIDDEN_SIZE // GROUP_SIZE   # 16
N_BLOCK = 4
SUBGROUP_SIZE = 32
SUBGROUP_NUM = 8

# Per-expert buffer element counts
EXPERT_WEI_SIZE   = INTERMEDIATE_SIZE * HIDDEN_SIZE // 2   # u4: N*K/2 bytes
EXPERT_SCALE_SIZE = INTERMEDIATE_SIZE * NUM_GROUPS          # f16: N*num_groups elements
EXPERT_ZP_SIZE    = INTERMEDIATE_SIZE * NUM_GROUPS // 2     # u4: N*num_groups/2 bytes


@pytest.fixture(scope="session")
def ocl_queue():
    """Fixture to create an OpenCL command queue for GPU execution."""
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


# ===================== Reference implementation =====================

def _dequantize_expert(weight_all, scale_all, zp_all, expert_id):
    """Dequantize u4 weights for one expert. Returns float32 array [N, K]."""
    N, K = INTERMEDIATE_SIZE, HIDDEN_SIZE

    # Weights: [N, K/2] row-major bytes; each byte = lo(k even) | hi(k odd)
    w_bytes = weight_all[expert_id * EXPERT_WEI_SIZE: (expert_id + 1) * EXPERT_WEI_SIZE]
    w_bytes = w_bytes.reshape(N, K // 2)
    w = np.empty((N, K), dtype=np.float32)
    w[:, 0::2] = (w_bytes & 0x0F).astype(np.float32)
    w[:, 1::2] = (w_bytes >> 4).astype(np.float32)

    # Scale: indexed as scale[n + group_idx * N] → stored as [num_groups, N]
    s_flat = scale_all[expert_id * EXPERT_SCALE_SIZE: (expert_id + 1) * EXPERT_SCALE_SIZE].astype(np.float32)
    s_mat = s_flat.reshape(NUM_GROUPS, N)           # [num_groups, N]
    scale_k = np.repeat(s_mat, GROUP_SIZE, axis=0).T  # [N, K]

    # ZP: indexed as zp[n//2 + group_idx * (N//2)] → stored as [num_groups, N/2] bytes
    # each byte: lo = even-n element, hi = odd-n element
    z_bytes = zp_all[expert_id * EXPERT_ZP_SIZE: (expert_id + 1) * EXPERT_ZP_SIZE]
    z_bytes = z_bytes.reshape(NUM_GROUPS, N // 2)
    zp_mat = np.empty((NUM_GROUPS, N), dtype=np.float32)
    zp_mat[:, 0::2] = (z_bytes & 0x0F).astype(np.float32)
    zp_mat[:, 1::2] = (z_bytes >> 4).astype(np.float32)
    zp_k = np.repeat(zp_mat, GROUP_SIZE, axis=0).T  # [N, K]

    return (w - zp_k) * scale_k  # [N, K]


def reference_gate_up(expert_list, gate_weight_all, gate_scale_all, gate_zp_all,
                      up_weight_all, up_scale_all, up_zp_all, x_f16):
    """Reference MoE gate+up projection with SwiGLU (f32 accumulation)."""
    x_f = x_f16.astype(np.float32)
    y = np.zeros(MAX_TOPK * INTERMEDIATE_SIZE, dtype=np.float16)

    for slot in range(MAX_TOPK):
        eid = int(expert_list[slot])
        dq_up   = _dequantize_expert(up_weight_all,   up_scale_all,   up_zp_all,   eid)
        dq_gate = _dequantize_expert(gate_weight_all, gate_scale_all, gate_zp_all, eid)

        up_out   = dq_up   @ x_f   # [N]
        gate_out = dq_gate @ x_f   # [N]

        # SwiGLU: result = up * swish(gate),  swish(x) = x / (1 + exp(-x))
        swish  = gate_out / (1.0 + np.exp(-gate_out.astype(np.float64))).astype(np.float32)
        result = up_out * swish

        y[slot * INTERMEDIATE_SIZE: (slot + 1) * INTERMEDIATE_SIZE] = result.astype(np.float16)

    return y  # float16, shape [MAX_TOPK * INTERMEDIATE_SIZE]


def _check_tolerance(result_f16, expected_f16):
    """Match the moe_test.cpp acceptance criterion: mismatch = rel_err>0.1 AND abs_err>0.5."""
    gpu = result_f16.astype(np.float32)
    ref = expected_f16.astype(np.float32)
    abs_err = np.abs(gpu - ref)
    rel_err = np.where(np.abs(ref) > 1e-6, abs_err / np.abs(ref), abs_err)
    num_mismatch = int(np.sum((rel_err > 0.1) & (abs_err > 0.5)))
    assert num_mismatch == 0, (
        f"{num_mismatch}/{result_f16.size} outputs exceed tolerance "
        f"(max_abs={abs_err.max():.4f}, max_rel={rel_err.max():.4f})"
    )


# ===================== Data generation =====================

def generate_test_data(seed=42):
    """Generate random u4/f16 MoE test data matching moe_test.cpp's scheme."""
    rng = np.random.default_rng(seed)

    expert_list = rng.choice(EXPERT_NUM, size=MAX_TOPK, replace=False).astype(np.int32)

    gate_weight = np.zeros(EXPERT_NUM * EXPERT_WEI_SIZE,   dtype=np.uint8)
    gate_scale  = np.zeros(EXPERT_NUM * EXPERT_SCALE_SIZE, dtype=np.float16)
    gate_zp     = np.zeros(EXPERT_NUM * EXPERT_ZP_SIZE,    dtype=np.uint8)
    up_weight   = np.zeros(EXPERT_NUM * EXPERT_WEI_SIZE,   dtype=np.uint8)
    up_scale    = np.zeros(EXPERT_NUM * EXPERT_SCALE_SIZE, dtype=np.float16)
    up_zp       = np.zeros(EXPERT_NUM * EXPERT_ZP_SIZE,    dtype=np.uint8)

    for eid in expert_list:
        eid = int(eid)
        ow, os_, oz = eid * EXPERT_WEI_SIZE, eid * EXPERT_SCALE_SIZE, eid * EXPERT_ZP_SIZE

        gate_weight[ow: ow + EXPERT_WEI_SIZE]     = rng.integers(0, 256, EXPERT_WEI_SIZE,   dtype=np.uint8)
        gate_scale[os_: os_ + EXPERT_SCALE_SIZE]  = rng.uniform(0.001, 0.1, EXPERT_SCALE_SIZE).astype(np.float16)
        gate_zp[oz: oz + EXPERT_ZP_SIZE]          = rng.integers(0, 256, EXPERT_ZP_SIZE,    dtype=np.uint8)

        up_weight[ow: ow + EXPERT_WEI_SIZE]       = rng.integers(0, 256, EXPERT_WEI_SIZE,   dtype=np.uint8)
        up_scale[os_: os_ + EXPERT_SCALE_SIZE]    = rng.uniform(0.001, 0.1, EXPERT_SCALE_SIZE).astype(np.float16)
        up_zp[oz: oz + EXPERT_ZP_SIZE]            = rng.integers(0, 256, EXPERT_ZP_SIZE,    dtype=np.uint8)

    x_f16 = rng.uniform(-1.0, 1.0, HIDDEN_SIZE).astype(np.float16)
    return expert_list, gate_weight, gate_scale, gate_zp, up_weight, up_scale, up_zp, x_f16


def get_data_on_device(ocl_queue_arg, seed=42):
    """Upload MoE test data to GPU and compute reference output."""
    data = generate_test_data(seed)
    expert_list, gate_weight, gate_scale, gate_zp, up_weight, up_scale, up_zp, x_f16 = data
    ctx = ocl_queue_arg.context

    def ro_buf(arr):
        return cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=arr)

    buf_expert_list = ro_buf(expert_list)
    buf_gate_weight = ro_buf(gate_weight)
    buf_gate_scale  = ro_buf(gate_scale)   # float16 → raw bytes
    buf_gate_zp     = ro_buf(gate_zp)
    buf_up_weight   = ro_buf(up_weight)
    buf_up_scale    = ro_buf(up_scale)     # float16 → raw bytes
    buf_up_zp       = ro_buf(up_zp)
    buf_x           = ro_buf(x_f16)        # float16 → raw bytes
    buf_y           = cl.Buffer(ctx, cl.mem_flags.READ_WRITE,
                                size=MAX_TOPK * INTERMEDIATE_SIZE * 2)  # float16 output

    args = (buf_expert_list, buf_gate_weight, buf_gate_scale, buf_gate_zp,
            buf_up_weight,   buf_up_scale,    buf_up_zp,       buf_x, buf_y)

    expected = reference_gate_up(expert_list, gate_weight, gate_scale, gate_zp,
                                 up_weight, up_scale, up_zp, x_f16)
    return args, expected


# ===================== Kernel initialization =====================

def initialize_moe_kernel(filename, queue):
    """Compile and return a callable MoE mlp_gate_up kernel."""
    cl_src = Path(__file__).parent / filename
    src_text = cl_src.read_text()

    build_options = []
    if filename == "moe_kernel.cl":
        # Evolved kernels may omit macro preamble (e.g. SUBGROUP_SIZE), which makes
        # attributes like intel_reqd_sub_group_size(SUBGROUP_SIZE) fail at compile time.
        # Inject stable defaults from this task via build options to keep the harness robust.
        build_options = [
            f"-DMAX_TOPK={MAX_TOPK}",
            f"-DEXPERT_NUM={EXPERT_NUM}",
            f"-DHIDDEN_SIZE={HIDDEN_SIZE}",
            f"-DINTERMEDIATE_SIZE={INTERMEDIATE_SIZE}",
            f"-DN_BLOCK={N_BLOCK}",
            f"-DSUBGROUP_SIZE={SUBGROUP_SIZE}",
            f"-DSUBGROUP_NUM={SUBGROUP_NUM}",
            f"-DGATE_UP_GROUP_SIZE={GROUP_SIZE}",
        ]

    program = cl.Program(queue.context, src_text).build(options=build_options)
    knl = program.mlp_gate_up

    # Dispatch matches moe_test.cpp:
    #   global: [MAX_TOPK, SUBGROUP_SIZE, INTERMEDIATE_SIZE/N_BLOCK] = [8, 32, 192]
    #   local:  [1, SUBGROUP_SIZE, SUBGROUP_NUM]                      = [1, 32,   8]
    g_size = (MAX_TOPK, SUBGROUP_SIZE, INTERMEDIATE_SIZE // N_BLOCK)
    l_size = (1, SUBGROUP_SIZE, SUBGROUP_NUM)

    def run(*args):
        return knl(queue, g_size, l_size, *args)

    return run


@pytest.fixture(scope="session")
def kernel(use_reference, ocl_queue):
    """Returns a callable kernel that runs the mlp_gate_up kernel on the GPU."""
    filename = "moe_reference.cl" if use_reference else "moe_kernel.cl"
    return initialize_moe_kernel(filename, ocl_queue)


# ===================== Test class =====================

class TestMoEDecodeOCL(CustomTest):
    """Custom task for MoE decode gate-up projection (mlp_gate_up) using pyopencl."""

    def build(self, gpu_arch) -> list[str]:
        """Validate compilation of moe_kernel.cl at build time."""
        device = cl.get_platforms()[0].get_devices()[0]
        ctx = cl.Context([device])
        build_queue = cl.CommandQueue(ctx)
        initialize_moe_kernel("moe_kernel.cl", build_queue)

    @pytest.mark.parametrize("_run", range(3))
    def test_correctness_wrt_pytorch(self, kernel, ocl_queue, _run):
        """Compare GPU kernel output against Python (f32) reference."""
        args, expected = get_data_on_device(ocl_queue, seed=_run)

        kernel(*args)

        buf_y = args[8]
        result = np.empty(MAX_TOPK * INTERMEDIATE_SIZE, dtype=np.float16)
        cl.enqueue_copy(ocl_queue, result, buf_y)

        _check_tolerance(result, expected)

    @pytest.mark.parametrize("_run", range(3))
    def test_correctness_wrt_reference(self, kernel, ocl_queue, _run):
        """Compare GPU kernel output against moe_reference.cl output."""
        args, _ = get_data_on_device(ocl_queue, seed=_run)

        kernel(*args)

        buf_y = args[8]
        result = np.empty(MAX_TOPK * INTERMEDIATE_SIZE, dtype=np.float16)
        cl.enqueue_copy(ocl_queue, result, buf_y)

        # Run reference OCL kernel into the same output buffer
        reference_fn = initialize_moe_kernel("moe_reference.cl", ocl_queue)
        reference_fn(*args)
        ref = np.empty(MAX_TOPK * INTERMEDIATE_SIZE, dtype=np.float16)
        cl.enqueue_copy(ocl_queue, ref, buf_y)

        _check_tolerance(result, ref)

    @pytest.mark.performance
    def test_benchmark(self, kernel, ocl_queue, measure_runtime):
        """Benchmark: time the GPU kernel dispatch only — no host<->device copies."""
        args, _ = get_data_on_device(ocl_queue, seed=0)
        measure_runtime(
            kernel,
            args=args,
            sync_fn=ocl_queue.finish,
            auto_replicate_inputs_size=0,
        )


if __name__ == "__main__":
    """Optional build when running this file directly during development."""
    task = TestMoEDecodeOCL()
    task.build(CustomTest.get_machine_gpu_arch())
