# How to extract an OpenVINO GPU micro-kernel into a standalone test

A field guide (method + pitfalls / 避坑) for taking a kernel from
`src/plugins/intel_gpu/src/graph/impls/ocl_v2/` that uses **oneDNN/gemmstone
microkernels** and rebuilding it as a self-contained C++ perf/correctness test
that runs **outside** OpenVINO.

Worked example: `sdpa_micro__generate` (the `micro_sdpa` decode kernel). The same
recipe applies to any micro-kernel-based op (gated MLP, other SDPA variants, MoE
GEMMs that use gemmstone, …).

> The single biggest time sink is a **stale static-library ABI mismatch**
> (section 4.1). Read that first if you hit a segfault/`bad_alloc` inside
> `selectGEMM`.

---

## 0. Why these kernels are special

A "micro-kernel" kernel is **not** a normal `.cl` file you can just compile.
Its inner GEMMs are generated as **machine code** by oneDNN's gemmstone JIT
(`ngen` emits Intel GPU ISA directly into memory), and that machine code is
**patched into the program binary after the first OpenCL compile**. So a
standalone test must reproduce three things the plugin normally hides:

1. **Host-side microkernel generation** — `selectGEMM` → `generateShim`
   (pure host C++, no GPU needed).
2. **Source assembly** — concatenate shims + JIT `#define`s + the `.cl` body in
   the exact order `kernels_cache.cpp` uses, with the `#include`d batch headers
   inlined.
3. **Build + fuse** — `clBuildProgram`, then `fuse()` to splice the microkernel
   ISA into the binary, then rebuild from that binary.

---

## 1. The pipeline (what the plugin does, what you must mirror)

```
            ┌──────────────── HOST (any machine, no GPU) ────────────────┐
 config  →  selectGEMM(KQ)  ─┐                                            │
 tables     selectGEMM(VS)  ─┤→  Package{binary, settings, grfMin, …}     │
            (gemmstone JIT)  │                                            │
                             └→ generateShim(...) → OpenCL-C shim strings  │
            ┌──────────────────────────────────────────────────────────┐ │
            │ assemble source:                                          │ │
            │   shim_kq + shim_vs            (the "jit" prefix)         │ │
            │ + KERNEL/FUNC macros + JIT #defines                       │ │
            │ + inlined batch headers (the kernel's #includes)          │ │
            │ + sdpa_micro.cl body          (the "str" body)           │ │
            └──────────────────────────────────────────────────────────┘ │
            └────────────────────────────────────────────────────────────┘
            ┌──────────────── GPU (needs XMX: BMG/PTL) ──────────────────┐
 clBuildProgram(source) → binary                                         │
 fuse(binary, source.c_str())        // splice microkernel ISA           │
 clCreateProgramWithBinary(binary) → clBuildProgram → clCreateKernel     │
 enqueue with GWS/LWS + args → compare vs f32 ref → benchmark            │
            └────────────────────────────────────────────────────────────┘
```

Reference sources in the tree:

| Concern | File |
|---|---|
| Kernel body | `.../ocl_v2/sdpa_micro.cl` (entry `micro_sdpa`) |
| Host generator | `.../ocl_v2/sdpa/sdpa_gen_micro.cpp` (`SDPAMicroGenerator`) |
| `selectGEMM`/`generateShim`/`fuse` wrapper | `.../kernel_selector/micro_utils.hpp` |
| gemmstone public headers | `.../thirdparty/onednn_gpu/src/gpu/intel/gemm/jit/include/gemmstone/` |
| Compile + **fuse** flow | `.../ocl_v2/.../kernels_cache.cpp` (`get_program_source`, `has_microkernels`) |
| `KERNEL()` etc. macros | `.../ocl_v2/utils/kernel_generator.cpp` (`make_base_jit_constants`) |
| Kernel `#include`s | `.../kernel_selector/cl_kernels/include/batch_headers/{generic_vector_ops,sdpa_utils,tile_ops}.cl` |

---

## 2. Step-by-step procedure

### 2.1 Locate the kernel and its host generator
Find the `.cl` entry point and the C++ class that configures it. For SDPA the
generator class encodes the variant in its constructor:
`SDPAMicroGenerator(prefill, gqa_single_token)` → `sdpa_micro__generate` is
`(false, false)`. Identify **which micro-GEMMs are actually built** for your
variant — e.g. the non-paged generate path uses only **KQ** (`K^T·Q`) and
**VS** (`V·S`); the `Kc`/`Vc` GEMMs exist only under
`#if IS_PAGED_ATTENTION && !IS_PREFILL`. Building GEMMs you don't need wastes
time and can fail to match the catalog.

### 2.2 Reverse-engineer the generator (transcribe, don't guess)
Read these methods and copy their logic 1:1 for your chosen variant:

- `init_microkernels` → the `GEMMProblem` / `GEMMOptions` / `SizeParams` /
  `StrategyRequirement` setup for each micro-GEMM.
- `get_jit_constants` → every `#define` (data types, `D_MAX`, `SUBGROUP_SIZE`,
  compression flags, strides, alignments…).
- `get_dispatch_data_func` → GWS/LWS and the scalar kernel args.
- `get_arguments_desc` → **exact argument order**.
- `get_build_options` → `clBuildProgram` flags.

### 2.3 Write a tiny compile/link PROBE *before* the full test
Do **not** write the 800-line test first. Write a ~40-line `probe_micro.cpp`
that just calls `selectGEMM` once and prints `package.binary.size()` /
`grfMin` / a `generateShim` length. This validates the **headers + link recipe +
ABI** in isolation. 90% of the pain (section 4.1) surfaces here, where it is
trivial to debug, instead of buried in the full test.

### 2.4 Build the microkernels (`selectGEMM`)
Value-initialize the problem (`GEMMProblem problem{};` — **mandatory**, see
4.2), set the data types, then specialize per GEMM. The KQ/VS skeletons for the
int8-asymmetric path:

```cpp
micro::GEMMProblem problem{};
problem.Ta_ext = micro::Type::s8;    // K/V int8
problem.Tb_ext = micro::Type::f16;   // Q f16
problem.Ta = problem.Tb = micro::Type::f16;
problem.Tc = problem.Tc_ext = micro::Type::f32;
problem.Ts = problem.Tc;

// KQ = K^T · Q
problem_kq.A.layout = micro::MatrixLayout::T;   // C.layout = T, B.layout = Pr
problem_kq.B.crosspack = 2;  problem_kq.B.tileR = d_max;  problem_kq.B.tileC = sg;
problem_kq.aqGroupM = 1;     problem_kq.aqGroupK = D;       // per-token scale/zp
sizes_kq = { .m=Lk, .n=Lq, .k=D, .batch=B*Hq };

// VS = V · S   (sizes derived from the KQ work-group tile!)
problem_vs.A.layout = micro::MatrixLayout::N;   // C.layout = N, B.layout = Pr
problem_vs.B.crosspack = 16;
problem_vs.aqGroupM = rnd_up_pow2(D);  problem_vs.aqGroupK = 1;
sizes_vs = { .m=D, .n=kq_wg_tile_n, .k=kq_wg_tile_m, .batch=B*Hq };
auto adjust_vs = [](micro::GEMMStrategy& s){ s.dpasw |= s.fused; };
```

Pass the tuned unroll/WG sizes as `StrategyRequirement`s and **wrap every
`selectGEMM` in try/catch** (4.9). Read back `getSetting("wg_tile_m"/"wg_tile_n"/
"sg_per_wg_m"/"sg_per_wg_n")` from the KQ package — you need them for both the VS
problem and the dispatch.

### 2.5 Generate shims and assemble the source
```cpp
ShimOptions o; o.subgroupSize = sg; o.useTileOps = true;
o.decorator = "kq";                       shim_kq = generateShim(gemm_kq, OpenCL_C, o);
o.microkernelID++; o.decorator = "vs";    shim_vs = generateShim(gemm_vs, OpenCL_C, o);
```
Then build the final string in **this order** (4.4):
`shim_kq + shim_vs` → manual macros (`KERNEL`, `OPTIONAL_SHAPE_INFO_ARG`, `FUNC`)
→ all JIT `#define`s → inlined batch headers (strip their `#include` lines) →
the `.cl` body. Add a `--dump-source` flag to write this out; it is your #1
debugging artifact.

### 2.6 Build + fuse
```cpp
program = clCreateProgramWithSource(ctx, 1, &src, &len, &e);
clBuildProgram(program, 1, &dev, build_opts, …);     // build_opts from get_build_options
clGetProgramInfo(program, CL_PROGRAM_BINARIES, …);   // pull the binary
gemmstone::microkernel::fuse(binary, source.c_str()); // splice microkernel ISA
program = clCreateProgramWithBinary(ctx, 1, &dev, …, &binary, …);
clBuildProgram(program, …);
kernel = clCreateKernel(program, "micro_sdpa", &e);
```

### 2.7 Dispatch + arguments
Mirror `get_dispatch_data_func` / `get_arguments_desc` exactly:
```
LWS = { sub_group_size, sg_per_wg, 1 }
GWS = LWS, then
      GWS[0] *= ceil_div(Lq, kq_wg_tile_n)
      GWS[1] *= Hq            // Q heads
      GWS[2] *= batch
scalar args: d = v_head_size, k = Lk, q = Lq
buffer args (compressed asym): K, Q, V, A(out), d, k, q,
             K_scales, K_zp, V_scales, V_zp
```

### 2.8 Reference + correctness + perf
Write an independent f32 reference that **mirrors the kernel exactly** (same
scale `1/sqrt(D)`, same dequant `(q - zp) * scale`, same causal rule if enabled).
Compare with a relative-L2 tolerance (~2e-2 for f16). Time `--iters` iterations
after warmup; report ms/iter, GFLOP/s, KV GB/s.

---

## 3. Build recipe (the exact flags)

Includes (point at the gemmstone source tree + the oneDNN build's generated
headers):
```
-I .../onednn_gpu/src/gpu/intel/gemm/jit
-I .../onednn_gpu_build/include
-I .../onednn_gpu/include
-I thirdparty/ocl/cl_headers
-I .../onednn_gpu/third_party
-I .../onednn_gpu/src
-I .../onednn_gpu/src/gpu/intel/jit/config
-I .../onednn_gpu/third_party/ngen
-I .../onednn_gpu/src/gpu/intel/gemm/jit/include
```
Defines (must match how the lib was built — especially `NDEBUG`, see 4.12):
```
-DCL_TARGET_OPENCL_VERSION=300 -DDNNL_ENABLE_CONCURRENT_EXEC
-DDNNL_ENABLE_CPU_ISA_HINTS -DDNNL_ENABLE_MAX_CPU_ISA -DDNNL_GPU_ISA_XE2
-DDNNL_X64=1 -DGEMMSTONE_BUILD_12HP -DGEMMSTONE_BUILD_12LP -DGEMMSTONE_BUILD_12P7
-DGEMMSTONE_BUILD_12P8 -DGEMMSTONE_BUILD_XE2 -DGEMMSTONE_BUILD_XE3
-DGEMMSTONE_BUILD_XE3P -DGEMMSTONE_CONFIG -DNGEN_CONFIG
-D__STDC_CONSTANT_MACROS -D__STDC_LIMIT_MACROS -DNDEBUG
```
Link (note the group + multiple-definition flags, see 4.1):
```
c++ -std=c++17 -fopenmp -fsigned-char <defines> <includes> test.cpp \
    -Wl,--allow-multiple-definition \
    -Wl,--start-group  libgemmjit_fresh.a  libopenvino_onednn_gpu.a  -Wl,--end-group \
    -lOpenCL -fopenmp -lpthread -ldl -o test
```
See `build_test.sh` for the complete, parameterized version.

---

## 4. Pitfalls & how to avoid them (避坑)

### 4.1 ⚠️ Stale static-library ABI mismatch — the #1 trap
**Symptom:** the probe compiles and links fine but **SIGSEGVs inside
`gemmstone::...::transpose()`** or throws **`std::bad_alloc` inside
`Generator<Core::XeN>::gemmMicrokernelPackage`** — i.e. crashes *inside* the
library, not in your code.

**Root cause:** `libopenvino_onednn_gpu.a` on disk was built from an **older
checkout** than the gemmstone **headers** you compile against. The `GEMMProblem`
struct layout drifts between checkouts (e.g. an `ngen::Product product` member
was inserted **before** the `std::vector binary` member). The prebuilt code then
reads `product`'s bytes as the vector's begin/end pointers → garbage → crash.

**How to confirm:** in gdb, the crashing frame reads the vector at
`this+0x1a0` while your current header places `binary` at `this+0x1b0`. A tiny
`size_probe.cpp` that prints `sizeof(GEMMProblem)` and member offsets under your
flags makes the drift obvious (it is **not** a C++11-vs-C++17 ABI issue — verify
that both std versions give identical offsets).

**Fix:** rebuild the gemmstone objects from the **current** tree and link them
**first**, on-demand:
```bash
make -C <onednn_gpu_build> -j"$(nproc)" \
     dnnl_gpu_intel_gemm_jit generatorXE2 generatorXE3 generatorXE3P
# pack ALL .o from those four CMake .dir folders:
ar qc libgemmjit_fresh.a <those .o>;  ranlib libgemmjit_fresh.a
# link inside --start-group, BEFORE the stale lib, with:
-Wl,--allow-multiple-definition -Wl,--start-group libgemmjit_fresh.a <stale>.a -Wl,--end-group
```
You must rebuild **both** `dnnl_gpu_intel_gemm_jit` (has `selectGEMM` /
`generateShim` / `fuse` / `transpose`) **and** the per-arch generators
(`generatorXE2/XE3/XE3P`, one big `.o` each — these contain
`Generator<Core::XeN>::gemmMicrokernelPackage`). Fixing only the first leaves the
second crash.

> On a **consistent** build (lib and headers from the same commit) the install
> `.a` alone is fine; the fresh-archive shim is only a workaround for a stale
> local build. Keep it behind a flag (`USE_FRESH_GEMM_JIT`).

### 4.2 `GEMMProblem` must be value-initialized
Use `micro::GEMMProblem problem{};` (braces). The addressing/quant fields have no
in-class defaults; a default-constructed-without-`{}` object feeds garbage to
`selectGEMM`.

### 4.3 The `KERNEL()` / `OPTIONAL_SHAPE_INFO_ARG` / `FUNC()` macros are missing
These are injected by `make_base_jit_constants()` / the `CodeBuilder` decoration
step in the plugin, **not** by the `.cl` file. A standalone build must define
them by hand:
`KERNEL(name) → __kernel void name`, `OPTIONAL_SHAPE_INFO_ARG → ""` (non-dynamic),
`FUNC(name) → name`, plus `KERNEL_ID`. Forgetting them yields cryptic
`clBuildProgram` syntax errors.

### 4.4 Source-assembly order is load-bearing
`kernels_cache.cpp::get_program_source` builds `jit + str + undefs`. The shims
are the **jit** prefix and **must come first**, before the JIT `#define`s and the
`.cl` body. Inline the kernel's `#include`s (strip the `#include "..."` lines and
prepend the header bodies); the three SDPA batch headers have no nested includes,
which keeps this simple.

### 4.5 `fuse()` must run **after** the first compile, then rebuild from binary
The order is: compile source → get binary → `fuse(binary, source.c_str())` →
`clCreateProgramWithBinary` → build again. `fuse(binary, const char* source)`
scans the source for microkernel sigils and splices the ISA. Skipping the rebuild
from the fused binary means the microkernel code is never actually present.

### 4.6 int8 KV must be **asymmetric** for this path
The kernel only declares `K_zp`/`V_zp` parameters when
`use_asymmetric_quantization` is on (they need `KEY_ATTR_ZP_DATA_T`). A
*symmetric* int8 config **won't even compile** because the host arg list and the
kernel signature diverge. Use per-token affine quant: `dequant = (q - zp)*scale`,
with `scale`/`zp` as `half`, one pair per token (`[B, Hkv, Lk]`).

### 4.7 Causal masking semantics differ for decode
The kernel's non-paged causal mask is `(k > q)` with **0-based** indices and no
history offset — correct for prefill, **wrong** for a decode step that attends
over a long history. Default causal **off** (full attention) for the generate
test, and if you do enable it, mirror the kernel's exact rule in the reference
(and remember the true query position is `history + iq`).

### 4.8 Architecture / `gmdid` detection
Read `gmdid` from `clGetDeviceInfo(CL_DEVICE_IP_VERSION_INTEL /*0x4250*/)`.
Detect XMX via the `cl_intel_subgroup_matrix_multiply_accumulate` extension.
Beware: **`ngen::Core` and OpenVINO's `gpu_arch` enums use different
numbering** (e.g. `ngen::Core::Xe2 = 8`, but `gpu_arch::xe2 = 7`). Don't mix
them. Representative gmdids for `--force-arch`: BMG/Xe2 `0x05004000`
(arch=20,release=1), PTL/Xe3 `0x07800000` (arch=30), Xe3p `0x07c00000` (arch=31).

### 4.9 `selectGEMM` throws — always try/catch
It raises `std::runtime_error("No matching kernel")` when the catalog has no
microkernel for your `(arch, types, sizes, requirements)`, and may `bad_alloc`
on a malformed `gmdid`. Wrap both calls and print `ex.what()` so an off-target
arch degrades gracefully instead of aborting.

### 4.10 Separate "host JIT" from "GPU execution"
Steps 2.4–2.5 are pure host code and run on **any** box (even a gen9 iGPU with no
XMX). Steps 2.6–2.8 need real XMX hardware (BMG/PTL). Add `--gen-only` (stop
after shim generation) and `--force-arch` (override the detected arch) so you can
**fully validate the generator locally** and only ship the execution to the
remote target. Expected local result for BMG/PTL h128 int8:
`KQ binary≈79456B grfMin=81 systolic=1 wg_tile=(256,16)`, `VS binary≈36576B`.

### 4.11 Catalog coverage is arch- and build-dependent
A given oneDNN checkout may only ship systolic int8 SDPA microkernels for some
arches (here: BMG/PTL work; Xe3p reports "No matching kernel"; XeHPC/PVC isn't
targeted). Don't assume an arch is broken just because `selectGEMM` declines —
check whether the catalog was built for it.

### 4.12 `NDEBUG` must match the library
gemmstone struct layout can depend on assertion macros. Build the test with the
**same** `-DNDEBUG` (and the same `GEMMSTONE_*` defines) the lib used; mismatched
assertion config reintroduces an ABI drift like 4.1. Crib the canonical flags
from the target's `flags.make` (`dnnl_gpu_intel_gemm_jit.dir/flags.make`).

### 4.13 Don't force-link every gemm-jit `.o` directly
Listing all objects on the link line (instead of as an archive) pulls in
`gen_kernel.cpp.o`, which needs `device_info_t::ngen_product` that isn't in the
archive → unresolved symbol. Pack them into a `.a` and let the linker pull only
what's used **on demand** inside `--start-group`.

### 4.14 KQ vs VS are not symmetric — copy the differences carefully
`A.layout`: KQ=`T`, VS=`N`. `C.layout`: KQ=`T`, VS=`N`. `B.crosspack`: KQ=`2`
(with `tileR=d_max`, `tileC=sg`), VS=`16`. `aqGroupM/K`: KQ=`(1, D)`,
VS=`(rnd_up_pow2(D), 1)`. VS sizes come from the **KQ** package's work-group tile
(`m=D, n=kq_wg_tile_n, k=kq_wg_tile_m`). VS also takes the
`strategy.dpasw |= strategy.fused` adjuster.

---

## 5. Reusable checklist for the next micro-kernel

1. [ ] Identify the `.cl` entry + generator class + the **exact variant** (which
       micro-GEMMs are built).
2. [ ] Transcribe `init_microkernels`, `get_jit_constants`,
       `get_dispatch_data_func`, `get_arguments_desc`, `get_build_options`.
3. [ ] Write a 40-line **probe** that calls `selectGEMM` once; get it to print a
       non-zero `binary.size()` **before** anything else.
4. [ ] If it crashes inside the lib → **stale-lib drift** (4.1): rebuild
       `dnnl_gpu_intel_gemm_jit` + per-arch generators, pack `*_fresh.a`, link
       first inside `--start-group` with `--allow-multiple-definition`.
5. [ ] Build both micro-GEMMs; read `getSetting(...)` for dispatch.
6. [ ] `generateShim` (unique `decorator`, bump `microkernelID`); assemble
       `shims + macros + #defines + inlined headers + body`; add `--dump-source`.
7. [ ] `clBuildProgram` → `fuse` → rebuild from binary → `clCreateKernel`.
8. [ ] Mirror GWS/LWS + arg order; write an f32 reference; compare + benchmark.
9. [ ] Add `--gen-only` / `--force-arch` to validate the host path off-target;
       document that execution needs XMX (BMG/PTL).

---

*Companion docs:* [info.md](info.md) (build/run/flags) ·
[remote_machine.md](remote_machine.md) (BMG/PTL execution + cliloader).
