# Remote execution (BMG / PTL with XMX)

The full `sdpa_micro_generate_test` (GPU build + run + benchmark) needs an Intel
GPU with **XMX / systolic** support. The dev box (Intel UHD 630, gen9) can only
exercise the **host-side** microkernel generation (`--gen-only`). Run the full
test on one of the remote targets below.

## Targets

- **PTL 12Xe GPU (Windows)** — same machine used by `moe_kernel_tests`:
    - Target hardware: PTL (Xe3)
    - GPU frequency: 2400 MHz
    - `Local_Admin@10.239.132.229`
    - password: `openvino`
    - cliloader: `C:\Users\Local_Admin\Downloads\clintercept-3.0.6-win64\Release\cliloader.exe`
    - suggested test dir: `D:\river\sdpa_micro`

- **BMG (Xe2)** — any box with a discrete BMG GPU and a recent Intel Compute
  Runtime (Level-Zero / OpenCL) plus oneAPI.

> Confirm the exact current host/credentials before connecting; the entry above
> mirrors `moe_kernel_tests/remote_machine.md` and may change.

## What to copy

This test links against the OpenVINO oneDNN-GPU static lib, so the simplest
flow is to **build on a machine that has the OpenVINO build tree** and copy the
resulting self-contained executable, *or* build on the remote target if it has
the same tree. You need:

- `sdpa_micro_generate_test.cpp`, `build_test.sh`
- the kernel sources it inlines (resolved via `--kernel-dir`, default is the
  in-tree `src/plugins/intel_gpu/src/graph/impls/ocl_v2/`):
  - `sdpa_micro.cl`
  - `../../../kernel_selector/cl_kernels/include/batch_headers/{generic_vector_ops,sdpa_utils,tile_ops}.cl`

If you copy only the binary to a machine without the source tree, pass
`--kernel-dir` pointing at a directory that contains `sdpa_micro.cl` and the
`include/batch_headers/` headers.

## Build on the target

### Linux (BMG)

```bash
cd .github/skills/dev_sdpa_micro_opt
./build_test.sh
```

Requires the OpenVINO Intel-GPU oneDNN component already built under
`build-x86_64-release` (see the *Stale-library note* in [info.md](info.md)).

### Windows (PTL)

Adapt `build_test.sh` to MSVC, or build the equivalent with the same flags:
include paths into `src/plugins/intel_gpu/thirdparty/onednn_gpu/...`, the
`GEMMSTONE_BUILD_*` / `NGEN_CONFIG` / `DNNL_*` defines, link
`openvino_onednn_gpu.lib` (+ the fresh gemm-jit objects) and `OpenCL.lib`.
The C++ source itself is portable (only `CL/cl.h` and the gemmstone headers).

## Run

```bash
# Default decode scenario: 1 token, 4096 history, D=128, 8 KV heads, int8 KV
./sdpa_micro_generate_test --tokens 1 --history 4096 --head-dim 128 --kv-heads 8 --heads 8 --iters 100

# GQA broadcast (8 Q heads share 2 KV heads):
./sdpa_micro_generate_test --kv-heads 2 --heads 8

# Causal:
./sdpa_micro_generate_test --causal 1
```

On a correct XMX target the kernel builds, the correctness check prints a PASS
(relative-L2 < 2e-2), and the benchmark reports ms/iter, GFLOP/s and KV GB/s.

## Profiling with cliloader (Windows / Intercept Layer for OpenCL)

```bat
set CLI_DevicePerformanceTiming=1
set CLI_DevicePerformanceTimeKernelInfoTracking=1
set CLI_DevicePerformanceTimeGWSTracking=1
set CLI_DevicePerformanceTimeLWSTracking=1
C:\Users\Local_Admin\Downloads\clintercept-3.0.6-win64\Release\cliloader.exe ^
    sdpa_micro_generate_test.exe --tokens 1 --history 4096 --iters 200
```

cliloader prints per-kernel device timing for `micro_sdpa` (after fusing, the
single dispatched kernel) so you can cross-check the in-test timing and inspect
the chosen GWS/LWS.

## Notes

- The microkernel ISA is generated **for the target arch** on the host. If you
  build on one machine and run on another, make sure the detected `gmdid` (via
  `CL_DEVICE_IP_VERSION_INTEL`) matches the intended arch, or use `--force-arch`
  to pin it.
- `--dump-source` writes the exact assembled OpenCL (`sdpa_micro_full.cl`) that
  is handed to `clBuildProgram`; keep it for debugging build failures on the
  target.
