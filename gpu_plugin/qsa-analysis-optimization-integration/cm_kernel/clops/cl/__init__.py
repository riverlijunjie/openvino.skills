"""Build and expose the clops C++ extension (csrc).

On first import this file runs a CMake build using g++.
The resulting shared library is cached in the `build/` subdirectory next to
this file, so subsequent imports are instant.

Set the environment variable DO_CMAKE=1 to force a full cmake reconfigure.
"""

import os
import subprocess
import sys

import pybind11

cwd = os.path.dirname(os.path.realpath(__file__))
build_path = os.path.join(cwd, "build")
dir_path = cwd

# pybind11 cmake directory
cmake_prefix = pybind11.get_cmake_dir()

btype = "RelWithDebInfo"
build_jobs = os.environ.get("CLOPS_BUILD_JOBS", "8")

cmake_need_config = not os.path.isfile(os.path.join(build_path, "CMakeCache.txt")) or int(
    os.environ.get("DO_CMAKE", "0")
)

if cmake_need_config:
    subprocess.run(
        [
            "cmake",
            "-B",
            build_path,
            "-S",
            dir_path,
            f"-DCMAKE_BUILD_TYPE={btype}",
            f"-DCMAKE_PREFIX_PATH={cmake_prefix}",
            # Pin both the legacy (FindPythonInterp/Libs) and modern (FindPython)
            # variable names to the interpreter actually running this build, not
            # whatever "python" resolves to first on PATH: pybind11's CMake macros
            # otherwise pick up an unrelated Python install (wrong ABI tag),
            # producing a .pyd/.so this interpreter cannot import.
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DPython_EXECUTABLE={sys.executable}",
            "-Wno-dev",
        ],
        shell=False,
        check=True,
    )

subprocess.run(
    ["cmake", "--build", build_path, "--config", btype, f"-j{build_jobs}"],
    shell=False,
    check=True,
)

from .csrc import *  # noqa: F401, F403

# ── helpers ──────────────────────────────────────────────────────────────────


def source(options=""):
    """Decorator that compiles a CM/OCL kernel from a function's docstring.

    Example::

        @cl.source(options="-cmc")
        def my_kernel():
            return r'''
            extern "C" _GENX_MAIN_ void add(...) { ... }
            '''
    """
    import inspect

    def _cl_kernel(f):
        frame = inspect.currentframe().f_back
        src_lines, line_no = inspect.getsourcelines(f)
        src = "\n" * (line_no + 1) + f()
        return kernels(src, options)

    return _cl_kernel
