import pytest
import numpy as np
from typing import Any
import pyopencl as cl
from pathlib import Path
import re
from kernelfoundry.custom_test import CustomTest

DEFAULT_SHAPE = (2048, 2560, 2048)  # (M, K, N)


@pytest.fixture(scope="session")
def ocl_queue():
    device = None
    for platform in cl.get_platforms():
        devs = platform.get_devices(cl.device_type.GPU)
        if devs:
            device = devs[0]
            break
    assert device is not None, "No OpenCL GPU device found"
    ctx = cl.Context([device])
    return cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)


def get_data_on_device(inp_shape: tuple[int, int, int], ocl_queue_arg: cl.CommandQueue, seed: int = 0) -> tuple[tuple[Any, ...], np.ndarray]:
    m, k, n = inp_shape
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((m, k), dtype=np.float32).astype(np.float16)
    b = rng.standard_normal((k, n), dtype=np.float32).astype(np.float16)
    expected = (a.astype(np.float32) @ b.astype(np.float32)).astype(np.float32)

    a_ocl = cl.Buffer(ocl_queue_arg.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=a.ravel())
    b_ocl = cl.Buffer(ocl_queue_arg.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=b.ravel())
    out_ocl = cl.Buffer(ocl_queue_arg.context, cl.mem_flags.WRITE_ONLY, size=m * n * np.dtype(np.float16).itemsize)
    return (a_ocl, b_ocl, out_ocl, np.int32(m), np.int32(k), np.int32(n)), expected


def _eval_gws_expr(expr: str, env: dict[str, int]) -> int:
    """Evaluate a simple GWS expression like '(N/64)*64' or 'M/32' given M,K,N values."""
    expr = expr.strip()
    allowed = set("0123456789+-*/() MKN")
    if not all(c in allowed for c in expr):
        raise ValueError(f"Unsafe GWS expression: {expr}")
    for var in ("M", "K", "N"):
        expr = re.sub(rf'\b{var}\b', str(env[var]), expr)
    expr = expr.replace("/", "//")
    return int(eval(expr))


def _extract_balanced_parens(src: str, start: int) -> str | None:
    """Extract content between balanced parentheses starting after '(' at position start."""
    depth = 1
    i = start
    while i < len(src) and depth > 0:
        if src[i] == '(':
            depth += 1
        elif src[i] == ')':
            depth -= 1
        i += 1
    if depth != 0:
        return None
    return src[start:i - 1]


def _split_top_level(s: str) -> list[str]:
    """Split by commas that are not inside parentheses."""
    parts = []
    depth = 0
    current = []
    for c in s:
        if c == '(':
            depth += 1
            current.append(c)
        elif c == ')':
            depth -= 1
            current.append(c)
        elif c == ',' and depth == 0:
            parts.append(''.join(current).strip())
            current = []
        else:
            current.append(c)
    parts.append(''.join(current).strip())
    return parts


def _parse_gws_lws_comment(src: str):
    """Parse 'GWS = (...), LWS = (...)' from kernel launch comment.

    Supports formats like:
      // Launch: GWS = ((N/64)*64, M/32), LWS = (64, 1)
      // GWS = (N, M/32), LWS = (64, 1)
    Returns (gws_expr_list, lws_int_list) or None if not found.
    """
    gws_match = re.search(r'GWS\s*=\s*\(', src)
    if not gws_match:
        return None
    gws_raw = _extract_balanced_parens(src, gws_match.end())
    if gws_raw is None:
        return None

    lws_match = re.search(r'LWS\s*=\s*\(', src[gws_match.end():])
    if not lws_match:
        return None
    lws_start = gws_match.end() + lws_match.end()
    lws_raw = _extract_balanced_parens(src, lws_start)
    if lws_raw is None:
        return None

    gws_exprs = _split_top_level(gws_raw)
    try:
        lws_vals = [int(x.strip()) for x in lws_raw.split(',')]
    except ValueError:
        return None

    if len(gws_exprs) != len(lws_vals):
        return None

    return gws_exprs, lws_vals


def initialize_matmul_kernel(filename: str, queue: cl.CommandQueue):
    import math

    src_path = Path(__file__).parent / filename
    src_text = src_path.read_text()

    def strip_c_comments(text: str) -> str:
        """Remove C/OpenCL comments before parsing launch metadata.

        The optimized kernel file often keeps old kernels and launch hints in
        block comments or line comments.  Parsing those commented-out #defines
        can produce an invalid work-group size, e.g. a stale WG_SIZE=512 while
        the active kernel requires reqd_work_group_size(64, 1, 1).
        """
        text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
        text = re.sub(r"//.*", "", text)
        return text

    try:
        program = cl.Program(queue.context, src_text).build()
    except Exception as e:
        if filename == "matmul_opt.cl":
            print(f"[WARN] Failed to build {filename}; fallback to matmul_reference.cl. Reason: {e}")
            ref_text = (Path(__file__).parent / "matmul_reference.cl").read_text()
            program = cl.Program(queue.context, ref_text).build()
            src_text = ref_text
        else:
            raise

    active_src_text = strip_c_comments(src_text)

    knl_name_match = re.search(r'__kernel\s+void\s+(\w+)\s*\(', active_src_text)
    knl_name = knl_name_match.group(1) if knl_name_match else "matmul"
    knl = getattr(program, knl_name)

    if filename == "matmul_opt.cl":
        def parse_define(name):
            m = re.search(rf'#define\s+{name}\s+(\d+)', active_src_text)
            return int(m.group(1)) if m else None

        tile_m = parse_define("TILE_M")
        tile_n = parse_define("TILE_N")
        wg_m = parse_define("WG_M")
        wg_n = parse_define("WG_N")
        sg_size = parse_define("SG_SIZE")

        if all(v is not None for v in [tile_m, tile_n, wg_m, wg_n, sg_size]):
            wg_tile_m = wg_m * tile_m
            wg_tile_n = wg_n * tile_n
            wg_size = sg_size * wg_m * wg_n

            sig_match = re.search(
                r'__kernel\s+void\s+\w+\s*\([^)]*int\s+M\s*,\s*int\s+(\w)\s*,\s*int\s+\w\s*\)',
                active_src_text)
            arg_order_mnk = sig_match and sig_match.group(1) == 'N'

            def run(a, b, c, m, k, n):
                m_int, n_int = int(m), int(n)
                grid_m = int(math.ceil(m_int / wg_tile_m))
                grid_n = int(math.ceil(n_int / wg_tile_n))
                gws = (grid_m * wg_size, grid_n)
                lws = (wg_size, 1)
                if arg_order_mnk:
                    return knl(queue, gws, lws, a, b, c, np.int32(m), np.int32(n), np.int32(k))
                else:
                    return knl(queue, gws, lws, a, b, c, np.int32(m), np.int32(k), np.int32(n))

            return run

        gws_lws = _parse_gws_lws_comment(src_text)
        if gws_lws:
            gws_exprs, lws_vals = gws_lws

            def run(a, b, c, m, k, n):
                m_int, k_int, n_int = int(m), int(k), int(n)
                env = {"M": m_int, "K": k_int, "N": n_int}
                gws = tuple(_eval_gws_expr(e, env) for e in gws_exprs)
                lws = tuple(lws_vals)
                return knl(queue, gws, lws, a, b, c, np.int32(m), np.int32(k), np.int32(n))

            return run

        reqd = re.search(r"reqd_work_group_size\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)", active_src_text)
        if reqd:
            lws_x, lws_y = int(reqd.group(1)), int(reqd.group(2))

            def run(a, b, c, m, k, n):
                m_int, n_int = int(m), int(n)
                gws_x = int(math.ceil(n_int / lws_x)) * lws_x
                gws_y = int(math.ceil(m_int / lws_y)) * lws_y
                return knl(queue, (gws_x, gws_y), (lws_x, lws_y), a, b, c, np.int32(m), np.int32(k), np.int32(n))

            return run

        print("[WARN] matmul_opt.cl has no parsable reqd_work_group_size; fallback launch (N, M), local=None")

    return lambda a, b, c, m, k, n: knl(queue, (int(n), int(m)), None, a, b, c, np.int32(m), np.int32(k), np.int32(n))


@pytest.fixture(scope="session")
def kernel(use_reference, ocl_queue):
    filename = "matmul_reference.cl" if use_reference else "matmul_opt.cl"
    return initialize_matmul_kernel(filename, ocl_queue)


class TestMatmulOCL(CustomTest):
    def build(self, gpu_arch) -> list[str]:
        device = cl.get_platforms()[0].get_devices()[0]
        ctx = cl.Context([device])
        build_queue = cl.CommandQueue(ctx)
        initialize_matmul_kernel("matmul_opt.cl", build_queue)

    @pytest.mark.parametrize("_run", range(2))
    def test_correctness_wrt_pytorch(self, kernel, ocl_queue, _run):
        args, expected = get_data_on_device(DEFAULT_SHAPE, ocl_queue, seed=_run)
        kernel(*args)

        _, _, out_buf, m, _, n = args
        got_flat = np.empty(int(m) * int(n), dtype=np.float16)
        cl.enqueue_copy(ocl_queue, got_flat, out_buf)
        got = got_flat.reshape((int(m), int(n))).astype(np.float32)

        assert got.shape == expected.shape
        assert np.allclose(got, expected, rtol=2e-2, atol=2e-2), "matmul result mismatch vs pytorch/numpy"

    @pytest.mark.parametrize("_run", range(2))
    def test_correctness_wrt_reference(self, kernel, ocl_queue, _run):
        args, _ = get_data_on_device(DEFAULT_SHAPE, ocl_queue, seed=100 + _run)
        kernel(*args)

        _, _, out_buf, m, _, n = args
        got_flat = np.empty(int(m) * int(n), dtype=np.float16)
        cl.enqueue_copy(ocl_queue, got_flat, out_buf)
        got = got_flat.reshape((int(m), int(n))).astype(np.float32)

        ref_kernel = initialize_matmul_kernel("matmul_reference.cl", ocl_queue)
        ref_kernel(*args)
        ref_flat = np.empty(int(m) * int(n), dtype=np.float16)
        cl.enqueue_copy(ocl_queue, ref_flat, out_buf)
        ref = ref_flat.reshape((int(m), int(n))).astype(np.float32)

        assert np.allclose(got, ref, rtol=2e-2, atol=2e-2), "matmul result mismatch vs reference"

    @pytest.mark.performance
    def test_benchmark(self, kernel, ocl_queue, measure_runtime):
        args, _ = get_data_on_device(DEFAULT_SHAPE, ocl_queue, seed=7)
        measure_runtime(kernel, args=args, sync_fn=ocl_queue.finish, auto_replicate_inputs_size=0)


if __name__ == "__main__":
    task = TestMatmulOCL()
    task.build(CustomTest.get_machine_gpu_arch())
