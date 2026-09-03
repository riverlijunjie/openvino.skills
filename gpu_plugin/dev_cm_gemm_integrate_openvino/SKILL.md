---
name: dev_cm_gemm_integrate_openvino
description: Integrate Intel C-for-Metal (CM) GEMM kernels into the OpenVINO GPU plugin as a new ImplementationManager — covers the full pipeline from KernelGenerator subclassing through KernelData/language setup, CM-native post-ops JIT, fused-op argument binding, build/install/test loop, and every correctness pitfall encountered.
---

# CM GEMM → OpenVINO GPU Plugin Integration Skill

Use this skill when plugging a standalone `.cm` kernel into the OpenVINO GPU
plugin framework (ocl_v2 / primitive_impl_ocl_base path). It documents every
non-obvious coupling between the plugin's kernel-lifecycle machinery and CM's
compile/dispatch requirements, and records the root-cause bugs that were found
and fixed during the GGUF GEMM integration.

This skill is **specific to the OV plugin integration layer**. For kernel
optimization, correctness, and tuning see `dev_cm_gemm_optimization`.

---

## 0. Integration timeline — what happened, in order (read this first)

This is the actual sequence the GGUF GEMM integration went through, end to
end. Use it as a map: each phase links to the section that has the full
detail, so you don't have to rediscover the same bugs in the same order.

1. **Wire up the `KernelGenerator`/`ImplementationManager` skeleton** for
   prefill (CM) + decode (existing OCL shuffle-GEMV) dual-path dispatch.
   Immediately hit a **segfault at `execute_stage`** because the generator
   inherited from `ocl::KernelGenerator`, which loads from the OCL sources DB
   and silently produces a null `stage->kernel` (§2, §3).
2. **Fixed the language/sources-DB mismatch** by overriding `get_kernel_data()`
   to force `KernelLanguage::CM` + the CM `SourcesDB` (§3), set
   `batch_compilation = false` (§4), and fix the entry point to the literal
   function name (§5). This got the kernel compiling and executing.
3. **Hit `CL_BUILD_PROGRAM_FAILURE` with an empty build log** once fused ops
   (post-ops) were wired through the stock OCL `FUSED_OPS` jitter — CM cannot
   compile OCL-only identifiers (`convert_half`, `eltwise0_input3`, ...).
   Designed a **CM-native post-ops JIT path** instead (§8) and fixed the
   fused-input argument slot indexing bug that followed (§9).
4. **Found a silent correctness bug**: Q4_1 in the OV plugin uses a
   **separate-plane** `pqs_T ‖ pd_T ‖ pm_T` weight layout, not the
   interleaved `pdm_T` layout a standalone/offline kernel might assume (§11).
5. **Cleaned up dead code** (an unused OCL post-processing generator class,
   stray includes/comments) once the CM-native post-ops path made it
   redundant (§16).
6. **Synced later standalone-kernel optimizations into the plugin copies**
   (e.g. `gemm_q40_full.cm`/`gemm_q41_full.cm` A_OUTER/A_BIG/A_DESC_TG tiling
   work done in the separate `cm_gguf_gemm_kernel/` benchmark repo) —
   established the rule: **merge hot-loop optimizations, but always keep the
   plugin-only ABI/JIT hooks** (`cm_output_t`, `CM_POST_OPS_DECLS`, layout
   comments) intact; do not blindly overwrite the plugin file with the
   standalone one.
7. **Root-caused a large performance regression reported after adding
   post-ops + f16 output**: NOT register spill. The post-ops epilogue used a
   **scalar per-lane loop nested inside an already-unrolled outer store
   loop**, producing up to ~1024 unrolled scalar ops and non-coalesced
   scalar memory reads per thread. Fixed by making `CM_APPLY_POST_OP_N` a
   **function-like, fully vectorized macro** (§19) — this is now the
   correct/required pattern; §8's post-ops description reflects it.
8. **Added optional fp16/fp32 output (and fp32 input) support** via
   `cm_output_t`/`CM_OUTPUT_F32` and `cm_input_t`/`CM_INPUT_F32` JIT
   typedefs, both in the plugin and later in the standalone kernel repo —
   measured **perf-neutral** (0.98-1.05x) on BMG, since these kernels are
   activation-load-latency bound, not store-bandwidth bound (§20).
9. **Root-caused a SECOND, much larger reported regression on the remote PTL
   (Xe3) box**: again NOT register spill (verified directly on hardware,
   §22) — the plugin's `pick_cm_gemm_config()` PTL branch had gone **stale**
   relative to later tuning done in the standalone benchmark repo. Fixing the
   dispatch config (not the kernel source) recovered ~1.19x end-to-end 1st
   token latency (§17, §21).

**Net lesson across steps 7 and 9**: when a regression is reported after a
change that "only added X", verify X on real hardware (register usage,
correctness, timing) before assuming X is the cause — twice in this project
the real cause was something adjacent (an unrelated stale dispatch config,
a latent scalar-loop antipattern) that happened to ship in the same commit.

---

## 1. Required files and their roles

| File | Role |
|---|---|
| `src/graph/impls/cm/gguf/fc_gguf_cm.hpp` | `ImplementationManager` subclass — `validate_impl`, `support_shapes`, `create_impl` |
| `src/graph/impls/cm/gguf/fc_gguf_cm.cpp` | `KernelGenerator` subclass(es) + `PrimitiveImplCM` subclass (`execute`, `get_arguments`, `clone`) |
| `src/graph/impls/cm/gguf/gemm_q*.cm` | CM kernel source files embedded in the binary via the codegen pipeline |
| `src/graph/impls/cm/CMakeLists.txt` | `file(GLOB_RECURSE KERNELS *.cm)` auto-picks up new `.cm` files |
| `src/graph/registry/fully_connected_impls.cpp` | Register your impl **before** the OCL fallback |
| `src/graph/impls/ocl_v2/gguf/fc_gguf_shuffle_gen.hpp` | Shared OCL shuffle-GEMV generators reused for the decode path |

---

## 2. Plugin kernel lifecycle — critical sequence

```
create_impl()          → FCGGUFCmImpl(node, params)
  add_stage(stage,p)   → stage->kd = stage->codegen->get_kernel_data(p)
                          (kd.code->str holds the source; kernel ptr is NULL here)
get_kernels_source()   → returns {stage->kd.code, ...}
kernels_cache.compile()→ compiles source, returns compiled_kernels map
set_kernels(compiled)  → stage->kernel = compiled_kernels[sub_kernel_idx]
execute()              → execute_stage(events, inst, stage)
                          (crashes if stage->kernel is still NULL)
```

**Consequence**: `stage->kernel` is `nullptr` after `add_stage` and until
`set_kernels` is called. Any access before that (e.g., in `get_kernel_name()`)
will crash. The display name `cm::fc_gguf_cm_` with an empty suffix is NORMAL
during the compile phase — it does not mean the kernel is broken.

---

## 3. `KernelGenerator` — the single most important pitfall

### MUST inherit from `cm::KernelGenerator`, NOT `ocl::KernelGenerator`

Both live under different paths and have the same interface, but the base
`get_kernel_data()` implementation differs critically:

| Base class | `kd.code->language` | `build_code()` source |
|---|---|---|
| `ocl::KernelGenerator` | `OCLC_V2` | OCL sources DB (`.cl` files) — `gemm_q41_full` **not found** |
| `cm::KernelGenerator` | `CM` | CM sources DB (`.cm` files) — correct |

If you use `ocl::KernelGenerator` (the default in the ocl_v2 world),
`SourcesDB::get_kernel_template("gemm_q41_full")` throws "OCL Kernel template
not found", `kd.code` is left null, and `get_kernels_source()` asserts/crashes.
The error is caught silently in `add_stage`, so the crash happens much later in
`execute_stage` when `stage->kernel` is null — extremely confusing to diagnose.

**Fix**: include `"../utils/kernels_db.hpp"` (CM SourcesDB header) and override
`get_kernel_data()` manually:

```cpp
[[nodiscard]] KernelData get_kernel_data(const RuntimeParams& params) const override {
    KernelData kd;
    kd.code = std::make_shared<KernelString>();
    kd.code->language    = KernelLanguage::CM;           // ← critical
    kd.code->entry_point = get_entry_point(params);
    kd.code->options     = get_build_options(params);
    kd.code->batch_compilation = false;  // ← see §4
    kd.code->has_microkernels  = false;
    // Prepend JIT #defines, append .cm source, then #undefs
    std::string src;
    for (const auto& c : get_jit_constants(params))
        src += "#define " + c.name + " " + c.value + "\n";
    src += std::string(ov::intel_gpu::cm::SourcesDB::get_kernel_template(get_kernel_name()));
    for (const auto& c : get_jit_constants(params))
        src += "#undef " + c.name + "\n";
    kd.code->str = std::move(src);
    kd.params.arguments         = get_arguments_desc(params);
    kd.params.layerID           = params.desc->id;
    kd.update_dispatch_data_func = get_dispatch_data_func();
    kd.need_args_update = kd.need_dispatch_data_update = true;
    return kd;
}
```

> **Note**: `cm::KernelGenerator` lacks helpers like `get_kernel_name()`,
> `make_base_jit_constants()`, etc. that live on `ocl::KernelGenerator`.
> Instead of inheriting from `cm::KernelGenerator`, inherit from
> `ocl::KernelGenerator` (so those helpers are available) and override
> `get_kernel_data()` manually as shown above. This is the approach used in
> `fc_gguf_cm.cpp`.

---

## 4. Batch compilation must be `false` for CM kernels with `#include`

`kernels_cache` groups kernels with the same build options into one OCL program
and concatenates their sources via `join_strings()`. If two CM kernels both
start with `#include <cm/cm.h>`, the combined source has a duplicate include.
Even with include guards this causes `CL_BUILD_PROGRAM_FAILURE` with an empty
build log (ocloc crashes silently).

**Fix**: always set `kd.code->batch_compilation = false` for CM kernels that
have their own `#include` directives. This gives each kernel its own unique
batch key (`__PROGRAM__N` suffix) and its own OCL program.

---

## 5. Entry point: use the literal function name

CM kernels use a literal `extern "C" _GENX_MAIN_ void kernel_name(...)`.
Override `get_entry_point()` to return exactly that name:

```cpp
[[nodiscard]] std::string get_entry_point(const RuntimeParams&) const override {
    return get_kernel_name();  // e.g. "gemm_q41_full"
}
```

The default base-class `get_entry_point()` appends a hash suffix (e.g.,
`gemm_q41_full_12345678__sa`) that will not match the literal function name in
the CM source, causing `kernels_cache` to throw "Could not find entry point"
after successful compilation.

---

## 6. Build options: never call the OCL base `get_build_options()`

`ocl::KernelGenerator::get_build_options()` returns OCL-specific flags
(e.g., `-cl-std=CL2.0 ...`). These are incompatible with `-cmc` mode and can
cause silent build failures. Always start from scratch:

```cpp
[[nodiscard]] std::string get_build_options(const RuntimeParams& params) const override {
    std::string opts = " -cmc";
    opts += " -DROW_GROUPS=2 -DTOKEN_GROUPS=4 ...";  // arch-specific
    return opts;
}
```

---

## 7. CM math functions — no standard `<math.h>` in the kernel

CM kernels compiled via `clBuildProgram` with `-cmc` do not have access to the
standard C math library. Replacements:

| Standard | CM equivalent |
|---|---|
| `expf(x)` | `cm_exp(x * 1.4426950408889634f)` (base-2 → natural) |
| `tanhf(x)` | `(cm_exp(2*x*1.4426f)-1)*cm_inv(cm_exp(2*x*1.4426f)+1)` |
| `erff(x)` | not directly available; use GELU tanh approximation instead |
| `native_exp(x)` | OCL only, unavailable in CM |
| `convert_half(x)` | OCL only; use `(half)x` in CM |
| `__global T*` | OCL only; use `T* ptr [[type("svmptr_t")]]` |

---

## 8. CM post-ops (fused ops): never reuse the OCL `FUSED_OPS` system

The OCL `make_fused_ops_jit_constants()` generates macros with OCL-specific
names (`eltwise0_input3`, `convert_half`, `__global`, etc.) that the CM
compiler will not accept. Building with those JIT constants causes
`CL_BUILD_PROGRAM_FAILURE` with errors like:

```
error: use of undeclared identifier 'convert_half'
error: use of undeclared identifier 'eltwise0_input3'; did you mean 'eltwise0_data3'?
```

### The CM-native post-ops pattern

Scan `params.fused_desc` directly and generate CM-compatible JIT macros:

```cpp
// In get_jit_constants():
// 1. Collect non-reorder ops and emit one CM_APPLY_POST_OP_N per op:
//    - swish: "v = v / (1.0f + cm_exp(-v * 1.4426950408889634f));"
//    - eltwise sum with outer dep: "v += (float)cm_post_in0[out_idx];"
//    - eltwise prod with outer dep: "v *= (float)cm_post_in0[out_idx];"
// 2. Emit CM_POST_OPS_NUM = count
// 3. Emit CM_POST_OPS_DECLS = ", half* cm_post_in0 [[type(\"svmptr_t\")]]"
//    for each outer-dep input
```

In the CM kernel — **`CM_APPLY_POST_OP_N` MUST be a function-like macro that
operates on the WHOLE output vector, never a scalar-per-lane statement** (see
§19 for why this is a hard requirement, not a style preference):

```c
#ifdef CM_POST_OPS_DECLS
    CM_POST_OPS_DECLS        // expands to ", half* cm_post_in0 [[type("svmptr_t")]]"
#endif

// Epilogue: ov = vector<float, N> accumulator (N = OPG or ROW_GROUPS*OPG).
// Apply post-ops to the WHOLE vector BEFORE narrowing to cm_output_t.
#if CM_POST_OPS_NUM > 0
{ const uint _po_base = tok * output_len + h0 * OPG;   // contiguous base, ELEMENT units
  #ifdef CM_APPLY_POST_OP_0
  CM_APPLY_POST_OP_0(ov, _po_base, N)   // macro args: (vector, base, width) -- all vectorized
  #endif
  #ifdef CM_APPLY_POST_OP_1
  CM_APPLY_POST_OP_1(ov, _po_base, N)
  #endif
}
#endif
vector<cm_output_t, N> ov_h = ov;   // narrow AFTER post-ops
```

`#if CM_POST_OPS_NUM > 0` when the macro is undefined → `#if 0` → block
disabled. No guard needed.

### Supported fused op types (generated statements are VECTORIZED, operate on `V`/`N` lanes at once)

| `fused_desc` primitive | Condition | Generated statement (operates on vector `V`, width `N`) |
|---|---|---|
| `cldnn::activation` / `swish` | — | `V = V / (1.0f + cm_exp(-V * 1.4426950408889634f));` (cm_exp accepts a vector arg) |
| `cldnn::activation` / `relu` | — | `V = cm_max<float>(V, 0.0f);` (**not** a ternary — see real usage in `pa_kv_cache_update_ref.cm`) |
| `cldnn::activation` / `gelu`/`gelu_tanh` | — | tanh approx via vector `cm_exp`/`cm_inv`, temporaries declared as `vector<float, N>` |
| `cldnn::eltwise` / `sum` | `has_outer_dep()` | ONE contiguous vector load of the fused operand (`cm_ptr_load<float,N>` or `cm_ptr_load<uint,N/2>().format<half>()` per its *actual* dtype, known at codegen time) then `V += _pin;` — **never** a per-lane `cm_post_inN[out_idx]` scalar load |
| `cldnn::eltwise` / `prod` | `has_outer_dep()` | same load, `V *= _pin;` |
| `cldnn::reorder` | any | **skip** — layout-only, never pass to kernel |

Return `false` from `support_shapes()` for any unsupported op type so the node
falls through to the OCL impl.

---

## 9. Fused-op argument binding: `outer_dep_start_idx` vs `fused_op_inputs`

`INPUT_OF_FUSED_PRIMITIVE` in `get_arguments_desc()` uses an **index into the
runtime `fused_op_inputs[]` array** (0-based, sequential over all outer deps).
`fd.outer_dep_start_idx` is the **absolute position in the node's full dep
list** (includes activation, weight, etc.).

Using `outer_dep_start_idx` directly as the index causes an out-of-bounds
assert at runtime:

```
Check 'args[i].index < data.fused_op_inputs.size()' failed
```

**Fix**: subtract `fused_mem_offset` (the minimum `outer_dep_start_idx` across
all fused ops) before using the value as the slot index:

```cpp
// Compute fused_mem_offset = min outer_dep_start_idx over all fused ops
int fused_mem_offset = -1;
for (const auto& fd : params.fused_desc)
    if (fd.outer_dep_start_idx >= 0)
        fused_mem_offset = (fused_mem_offset < 0)
            ? fd.outer_dep_start_idx
            : std::min(fused_mem_offset, fd.outer_dep_start_idx);

// Then for each non-reorder fused op with outer dep:
const auto slot = static_cast<uint32_t>(fd.outer_dep_start_idx - fused_mem_offset);
args.push_back({ArgumentDescriptor::Types::INPUT_OF_FUSED_PRIMITIVE, slot});
```

---

## 10. `get_arguments()` in `PrimitiveImplCM`

The runtime call `get_arguments(instance)` must populate `fused_op_inputs` for
ALL outer deps (including reorder scale tensors), even though the kernel only
uses the non-reorder ones. The runtime machinery expects the full list to be
present so the slot indices computed in §9 resolve correctly:

```cpp
if (instance.has_fused_primitives()) {
    const size_t count = instance.get_fused_mem_count();
    for (size_t i = 0; i < count; ++i)
        data.fused_op_inputs.push_back(instance.fused_memory(i));
}
```

---

## 11. Weight format: GGUF Q4_1 small-block layout correctness

The OV GPU `RepackGGUFWeightsShuffle` transform produces **identical layout for
ALL small-block formats (Q4_0, Q4_1, Q8_0)**:

```
weights = pqs_T || pd_T || pm_T
  pqs_T: num_blocks * OPG * 128 B  (nibble payloads, SG-transposed)
  pd_T:  num_blocks * OPG *  16 B  (scale d,  fp16 per lane)
  pm_T:  num_blocks * OPG *  16 B  (offset m, fp16 per lane)
```

Scale and offset are stored in **two separate planes**, 32 B each per block.

A standalone CM kernel developed outside OV GPU may assume Q4_1 packs d and m
**interleaved** (64 B per block, one load each). That assumption is WRONG in
the OV context and produces silently incorrect GEMM output. The fix is to read
two separate 32 B loads exactly like Q4_0:

```c
// WRONG (standalone / pre-fix):
uchar* pdm_base = weights + num_blocks * 128u;  // interleaved
vector<uint, OPG> pdm_u = cm_ptr_load<uint, OPG>(pdm_base, blk*(OPG*32u)+j*64u);
vector<float, OPG> dF = pdm_u.format<half>().select<OPG,2>(0);
vector<float, OPG> mF = pdm_u.format<half>().select<OPG,2>(1);

// CORRECT (matches RepackGGUFWeightsShuffle):
uchar* pd_base = weights + num_blocks * 128u;
uchar* pm_base = pd_base + num_blocks * 16u;
dF = cm_ptr_load<uint, OPG/2>(pd_base, blk*(OPG*16u)+j*32u).format<half>();
mF = cm_ptr_load<uint, OPG/2>(pm_base, blk*(OPG*16u)+j*32u).format<half>();
```

---

## 12. Support shapes: what to check

Implement `support_shapes()` to decline nodes the CM impl cannot handle:

```cpp
bool support_shapes(const kernel_impl_params& params) const override {
    const auto& in1 = params.get_input_layout(1);  // weight
    // 1. Weight must be static with supported GGUF format
    if (in1.is_dynamic()) return false;
    if (!is_supported_format(in1.data_type)) return false;
    // 2. Shuffle layout must not be disabled
    if (const char* e = std::getenv("OV_GPU_GGUF_SHUFFLE"))
        if (std::atol(e) == 0) return false;
    // 3. Geometry: N%32==0, K%256==0 (SG-shuffle layout constraint)
    if (in1.get_shape()[0] % 32 != 0 || in1.get_shape()[1] % 256 != 0) return false;
    // 4. Output must be fp16 (CM GEMM accumulates fp32, stores fp16)
    const auto& out = params.get_output_layout(0);
    if (!out.is_dynamic() && out.data_type != data_types::f16) return false;
    // 5. All non-reorder fused ops must be CM-supported types
    if (params.has_fused_primitives())
        for (const auto& fd : params.fused_desc) {
            if (fd.is_type<cldnn::reorder>()) continue;
            if (fd.is_type<cldnn::activation>()) continue;
            if (fd.is_type<cldnn::eltwise>())   continue;
            return false;  // unknown op → fall through to OCL
        }
    return true;
}
```

---

## 13. Registry order matters

Register your CM impl **before** the OCL fallback in
`fully_connected_impls.cpp`. The registry is searched in order; the first
matching impl wins:

```cpp
OV_GPU_CREATE_INSTANCE_CM(cm::FCGGUFCm, shape_types::any)   // ← before OCL
OV_GPU_CREATE_INSTANCE_OCL(ocl::FCGGUFOpt, shape_types::any)
```

---

## 14. Build / install / test loop

```bash
# Build (source tree build dir):
cd build-x86_64-release
make -j28 openvino_intel_gpu_plugin

# Install (CRITICAL — run_gguf.sh sources install_release/setupvars.sh,
# NOT the build dir; without this step the test uses the OLD binary):
cp bin/intel64/Release/libopenvino_intel_gpu_plugin.so \
   install_release/runtime/lib/intel64/libopenvino_intel_gpu_plugin.so

# Clear NEO compiler cache (CM kernels are compiled at runtime; cache holds
# stale binaries from previous runs and masks source changes):
rm -rf ~/.cache/neo_compiler_cache/*

# Run test:
cd .../llm_bench
source .../install_release/setupvars.sh
source .../venv/bin/activate
OV_VERBOSE=4 python3 benchmark.py -m .../Qwen3-8B-Q4_1.gguf -d GPU --genai \
    -n 1 -ic 8 -pf .../1024/qwen3-8b.jsonl
```

Forgetting `cp .so` or `rm -rf neo_cache` are the two most common reasons a
change appears to have no effect or fails with a stale error.

---

## 15. Diagnosing kernel build failures

### Empty build log: `CL_BUILD_PROGRAM_FAILURE (-11)` with nothing between markers

```
GPU_Debug: ocl_kernel_builder.hpp:67: -------- Kernel build error
GPU_Debug: ocl_kernel_builder.hpp:72: -------- End of Kernel build error
```

The build log is printed via `GPU_DEBUG_INFO`. Even with `OV_VERBOSE=4` it can
be invisible (compiled out in some configs). Add a temporary `std::cerr` in the
`catch (const cl::BuildError&)` block to surface the ocloc output:

```cpp
for (auto& e : log) {
    GPU_DEBUG_INFO << e.second;
    std::cerr << "[CM_DBG] " << e.second << std::endl;  // temp
}
```

Remove after debugging.

### `"OCL Kernel template X not found"` in `add_stage` log

Root cause: inheriting from `ocl::KernelGenerator` instead of CM (see §3).

### `"CM_POST_OPS_NUM > 0"` — unexpected compilation failure

Check that `CM_APPLY_POST_OP_N` macros use only CM-available functions (no
`native_exp`, `convert_half`, `tanhf`). See §7.

### `"Could not find entry point"` after successful compile

`get_entry_point()` is returning a hash-suffixed name that doesn't match the
literal function name in the `.cm` file. Override it (see §5).

### Segfault at `execute_stage` with `stage->kernel == nullptr`

Either `get_kernel_data()` threw (check `add_stage: Failed to get kernel data`
in log), or `set_kernels()` received zero compiled kernels (batch compilation
failure, or entry-point mismatch). Never call `execute_stage` without first
verifying `stage->kernel != nullptr`.

---

## 16. Dead code to avoid

When standing up a new CM impl, do NOT add:

- A separate `FCGGUFCmFusedOpsGenerator` OCL post-processing class — fused ops
  belong inline in the CM kernel via `CM_APPLY_POST_OP_N` macros (§8).
- `make_base_jit_constants()`, `make_tensors_jit_constants()`,
  `make_fused_ops_jit_constants()` in a CM-path generator — these generate
  OCL-incompatible code.
- `#include <cassert>`, `<mutex>`, `<unordered_map>` unless actually used.
- A `// Must match gguf_shuffle_applicable()` comment pointing at a function
  that may not exist or may not be the actual constraint.

---

## 17. Architecture config dispatch (BMG vs PTL)

Architecture-specific tile config belongs in a `pick_cm_gemm_config()` helper:

```cpp
CmGemmConfig pick_cm_gemm_config(const RuntimeParams& params, size_t K) {
    const bool is_xe3 = (params.get_device_info().arch >= gpu_arch::xe3);
    (void)K;  // no PTL long-K promotion needed with the RG=4 tile below
    if (is_xe3) {
        // PTL (Arc B390, Xe3): ADOPTED tile, matches the standalone
        // cm_gguf_gemm_kernel/ ab_lowbit_config.py "PTL round 2" result
        // (measured 1.01-1.63x over the older RG=2/RB=2/TL=32 tile, across
        // all six quants). Do NOT re-add a "K>=10240 -> TOKEN_GROUPS=4"
        // promotion on top of RG=4 -- that gives 8 KB of accumulators and
        // spills (0.19-0.27x); RG=4 already wins at long K directly.
        return {/*row_groups*/4, /*token_groups*/2, /*token_local*/16,
                /*row_blocks*/1, /*slm_jblk*/4, /*pf_a*/1, /*a_outer*/1};
    }
    // BMG (Xe2): RG=2 TG=4 TL=8 RB=1 A_OUTER=1
    return {2, 4, 8, 1, /*slm_jblk*/-1, /*pf_a*/-1, 1};
}
```

Pass `-DROW_GROUPS=N`, `-DTOKEN_GROUPS=N`, etc. as `-D` flags in
`get_build_options()`. These are processed before the JIT `#define` block,
so they can be used in kernel code as regular macros.

> **Config staleness trap** (see §21 for the full case study): this function
> is the ONLY place per-architecture tuning lives in the plugin. If the
> standalone benchmark repo (`cm_gguf_gemm_kernel/`) later discovers a better
> tile for an architecture, that improvement does **not** automatically apply
> to the plugin — someone has to port the new numbers into this function.
> A stale config here silently costs 1.07-1.43x (measured) and looks exactly
> like a "the last change I made caused a regression" bug even when the last
> change (e.g. post-ops) is completely unrelated and innocent.

---

## 19. Post-ops MUST be vectorized — a scalar-per-lane loop caused a large, hard-to-diagnose regression

**Symptom**: after adding post-ops + optional f16 output to the plugin's
`gemm_q*_full.cm` kernels, both a local (BMG) and a remote (PTL) test showed
"the kernel got a lot slower", with the diff between versions showing nothing
but "added post-ops" and "changed output to f16" — a strong, but wrong,
temptation to blame register spill.

**Root cause**: the ORIGINAL `CM_APPLY_POST_OP_N` implementation was a plain
(non-function-like) macro operating on a SCALAR `v`/`out_idx`, invoked inside
a `#pragma unroll for (int _j = 0; _j < N; _j++)` loop (N = OPG=16 or
ROW_GROUPS*OPG=32) **nested inside the already-unrolled per-token epilogue
store loop** (up to TOKEN_GROUPS*TOKENS_PER_TILE = 32 stores/thread). With any
real fused post-op present (e.g. a residual eltwise sum — common on FC
layers), this expanded to up to **~1024 unrolled scalar float ops per
thread**, PLUS N individual **scalar, non-coalesced** loads of the fused
operand (`cm_post_inN[out_idx]`) instead of one contiguous vector load.
Massive code-size bloat and scalar memory traffic — NOT the hot K-loop / dpas
path, which was completely untouched.

**Fix**: made `CM_APPLY_POST_OP_N` a **function-like macro** `(V, BASE, N)`
that mutates the WHOLE output vector `V` in 2-3 vector instructions (see §8
for the exact pattern):
- Activation ops (swish/relu/gelu) operate on the full vector directly
  (`cm_exp`/`cm_inv` accept vector args and deduce width from the argument;
  relu is `cm_max<float>(V, 0.0f)`, not a ternary).
- Eltwise sum/prod issue ONE contiguous vector load of the fused operand,
  typed per its actual runtime dtype (known at codegen time), then one
  vector `+=`/`*=`.
- The codegen side (`CmKernelGeneratorBase::get_kernel_data()`) must also
  strip the macro's parameter list before emitting `#undef`, since
  `#undef FOO(a,b)` is invalid — only `#undef FOO` is legal:
  ```cpp
  const auto paren = c.name.find('(');
  src += "#undef " + c.name.substr(0, paren) + "\n";
  ```

**How this was confirmed on real hardware, not just by inspection**: compiled
the actual kernel via pyopencl with `-mCM_printregusage` (works when invoked
directly through pyopencl, even on the box where it's silently ignored via
other tooling paths) and `PYOPENCL_NO_CACHE=1` (forces a real recompile
instead of a cached hit), with and without a representative post-op. Register
usage was **identical** with/without post-ops in the actual dispatched config
— i.e. the vectorized fix is genuinely free, and the scalar-loop version's
slowdown was never really about registers at all, it was about instruction
count and memory access pattern.

**Rule going forward**: when generating CM JIT epilogue/post-op code from
C++, always operate on the whole vector via a parametrized macro. Never nest
a `#pragma unroll for` scalar-lane loop inside an already-unrolled outer
loop — the multiplicative unrolling is easy to miss in review and causes
order-of-magnitude code bloat, not just a few extra instructions.

---

## 20. Optional fp16/fp32 output and input via JIT typedefs

The GEMM kernels accumulate in fp32 registers regardless of output dtype;
output/input element type is selected by a JIT-defined typedef, controlled by
a build-option macro, so the SAME `.cm` source serves both dtypes:

```c
#ifdef CM_OUTPUT_F32
typedef float cm_output_t;
#else
typedef half cm_output_t;      // default: half the store bandwidth/messages
#endif

#ifdef CM_INPUT_F32
typedef float cm_input_t;
#else
typedef half cm_input_t;        // default: matches the systolic dpas operand type
#endif

extern "C" _GENX_MAIN_ void gemm_q41_full(
    cm_input_t*  inputs  [[type("svmptr_t")]],
    uchar*       weights [[type("svmptr_t")]],
    cm_output_t* outputs [[type("svmptr_t")]],
    uint token_len, uint input_len, uint output_len
#ifdef CM_POST_OPS_DECLS
    CM_POST_OPS_DECLS
#endif
)
```

Epilogue store, narrowed AFTER post-ops (§19):

```c
vector<cm_output_t, N> ov_h = ov;   // ov is the float accumulator vector
#ifdef CM_OUTPUT_F32
cm_ptr_store<float, N>((float*)outputs, byte_off, ov_h);
#else
cm_ptr_store<uint, N/2>((uint*)outputs, byte_off, ov_h.format<uint>());  // half packed as uint
#endif
```

`get_build_options()` sets `-DCM_OUTPUT_F32=1` only when
`params.get_output_layout(0).data_type == data_types::f32`. The plugin's real
activation input is ALWAYS fp16 by the time the GEMM kernel runs (an fp32
activation is pre-converted by a small separate kernel,
`convert_act_f32_to_f16.cm`, before the big GEMM dispatch), so
`cm_input_t`/`CM_INPUT_F32` in the GEMM kernels themselves is defensive
support, not something the plugin actually exercises today.

**Measured**: this is **perf-neutral** (0.98-1.05x on BMG, interleaved
min-of-N across all six quants and four representative (K,N) shapes) — these
kernels are activation-load-latency bound, not store-bandwidth bound, so
halving the output store bytes doesn't move the needle. Worth keeping as the
default (halves memory footprint for downstream fp16 consumers) but don't
expect it to speed up the GEMM itself.

---

## 21. Keep the architecture-tuned dispatch config in sync with the standalone benchmark repo

See §17 for the `pick_cm_gemm_config()` pattern. This case study is why the
warning box there exists.

**What happened**: the standalone `cm_gguf_gemm_kernel/` benchmark repo went
through several rounds of PTL (Xe3/Arc B390) tuning across separate sessions
and eventually adopted `ROW_GROUPS=4 ROW_BLOCKS=1 TOKEN_LOCAL=16 SLM_JBLK=4
PF_A=1` (measured 1.01-1.63x faster than an older tile, across all six
quants, via `ab_lowbit_config.py`'s interleaved min-of-N methodology). The
plugin's `pick_cm_gemm_config()` Xe3 branch, however, still shipped the OLDER
`ROW_GROUPS=2 ROW_BLOCKS=2 TOKEN_LOCAL=32 SLM_JBLK=8 PF_A=0` tile — the
standalone repo's improvement was simply never ported over.

This surfaced as a large, confusing "1st token latency regressed a lot"
report on the remote PTL box right after an UNRELATED post-ops change, making
post-ops the natural (but wrong) suspect.

**How it was found**: don't trust a source diff alone to tell you "nothing
else changed" — a stale *runtime dispatch config* produces zero source diff
noise in the kernel file itself (the `.cm` source is identical; only the `-D`
values passed by `get_build_options()` differ). Confirmed by:
1. Checking the remote checkout's git commit hash matched local exactly, and
   the plugin DLL's mtime was newer than the source mtime (ruled out stale
   build).
2. Directly A/B'ing the SAME kernel binary logic on the real PTL device,
   varying ONLY the `-D` dispatch macros (old vs adopted config), interleaved
   in one process: 1.07-1.43x speedup from the config change alone, across 4
   representative shapes — this isolated the effect completely from
   post-ops (which weren't even in this A/B).
3. Applying the fix and re-running the full E2E benchmark end to end:
   1st token latency 1079 ms → 908 ms (~1.19x), consistent with the isolated
   kernel-level measurement.

**Rule going forward**: whenever the standalone benchmark repo lands a new
"ADOPTED" tuning result for an architecture, immediately check whether
`pick_cm_gemm_config()` in the plugin needs the same update. Treat the
plugin's tuning table as a second, easily-forgotten copy that has to be
manually kept in sync — there is no automation linking the two.

---

## 22. Validating on real remote hardware (register spill diagnosis + Windows box gotchas)

When a report says "kernel X regressed a lot" and register spill is suspected,
**don't guess from source inspection** — cross-compile the exact kernel with
the exact dispatch macros on the real target device and read the compiler's
own register-usage report. This is fast (seconds) and definitive.

```python
import os
os.environ["PYOPENCL_NO_CACHE"] = "1"        # force a REAL recompile, not a cache hit
os.environ["PYOPENCL_COMPILER_OUTPUT"] = "1" # surface compiler warnings via python warnings

import pyopencl as cl
ctx = cl.create_some_context(interactive=False)
src = open(r"gemm_q41_full.cm").read()
opts = "-cmc -mCM_printregusage -DROW_GROUPS=... -DTOKEN_GROUPS=... ..."
prog = cl.Program(ctx, src).build(options=opts)   # register-usage line goes to stderr
```

The `-mCM_printregusage` line (e.g. `Kernel gemm_q41_full : 102 registers`)
and a `Spill memory used = N bytes for kernel ...` line (if present) appear on
stderr from the driver, not in `prog.get_build_info(dev,
cl.program_build_info.LOG)` — capture raw stderr, don't rely on the build log
string being non-empty.

### Remote Windows (PTL) box access, via `/tmp/rptl.py` (paramiko helper)

Credentials + paths live in `cm_gguf_gemm_kernel/remote_machine.txt` (never
print them). Recreate the helper if `/tmp` is wiped — it wraps commands as
`cmd /c "call <venv_activate>.bat && cd /d <workdir> && <cmd>"`.

Gotchas that cost real debugging time:
- The `raw` subcommand does **NOT** `cd`/activate the venv — always use full
  `D:\...` absolute paths with it, or use `run` (which does wrap) instead.
- The remote default shell is `cmd.exe`. Plain `cd D:\other\drive\path` does
  **NOT** switch drives in cmd.exe — you must use `cd /d D:\other\drive\path`,
  otherwise the command silently runs in the SSH session's default directory
  (typically the Windows user's home dir) and file-not-found errors look like
  a missing-repo problem instead of a `cd` mistake.
- `findstr` on a UTF-16 (Unicode) log file prints a warning and often matches
  nothing usefully; read the file with a small Python helper
  (`io.open(path, encoding='utf-16')`) uploaded via `rptl.py py` instead of
  fighting `findstr`/PowerShell quoting through nested SSH `exec_command`
  layers (bash `$_`/`&&` expansion inside nested quotes is a frequent trap).
- Bash's `$_` special parameter will silently mangle any nested
  `powershell -Command "... $_.Something ..."` string passed through a
  double-quoted bash command — prefer a small uploaded Python script over
  ad-hoc nested-quoted one-liners for anything non-trivial on the remote box.

---

## 23. Key lessons summary

1. **`language = KernelLanguage::CM` is mandatory** — it controls which
   compiler (`ocloc -cmc`) and which sources DB (`.cm` files) are used.

2. **`batch_compilation = false`** for any CM kernel with `#include` at the
   top; otherwise multiple kernels get concatenated and the duplicate
   `#include <cm/cm.h>` breaks compilation silently.

3. **Entry point = literal function name**; do not use the hash-suffixed
   default from the OCL base class.

4. **Never call the OCL base `get_build_options()`** from a CM generator; it
   adds OCL flags incompatible with `-cmc`.

5. **Post-ops need a CM-native JIT path**; the OCL `FUSED_OPS` system uses
   OCL-only identifiers and will not compile under `-cmc`.

6. **`INPUT_OF_FUSED_PRIMITIVE` index = `outer_dep_start_idx − fused_mem_offset`**,
   not the raw `outer_dep_start_idx`. Using the raw value causes an
   out-of-bounds assert at runtime.

7. **Q4_1 weight layout in OV** = `pqs_T ‖ pd_T ‖ pm_T` (two separate 16 B
   planes), the same as Q4_0/Q8_0 — NOT the interleaved 32 B `pdm_T` layout
   a standalone kernel might assume.

8. **Always `cp .so` to install and `rm -rf neo_cache`** between builds;
   without both steps the running binary may be stale.

9. **CM math**: use `cm_exp(x * log2e)` for `exp(x)` and `cm_inv(x)` for
   `1/x`; `expf`/`tanhf`/`native_exp`/`convert_half` do not exist in CM.

10. **Dead code trap**: a separately designed OCL post-processing generator
    class for fused ops is architecturally redundant once CM kernels handle
    fused ops inline; remove it to avoid confusion and stale OCL-path JIT
    calls.

11. **Post-ops must be vectorized, never scalar-per-lane** (§19). A scalar
    `#pragma unroll for` loop nested inside an already-unrolled epilogue
    store loop caused up to ~1024 unrolled scalar ops + non-coalesced scalar
    memory reads per thread — a large regression that looked exactly like
    register spill but wasn't (confirmed identical register counts with/
    without post-ops via `-mCM_printregusage` on real hardware).

12. **fp16/fp32 output and input are cheap, JIT-selected typedefs**
    (`cm_output_t`/`CM_OUTPUT_F32`, `cm_input_t`/`CM_INPUT_F32`, §20) and are
    perf-neutral for these activation-load-latency-bound kernels — don't
    expect a store-width change to speed up the GEMM itself.

13. **Architecture-tuned dispatch config (`pick_cm_gemm_config()`) can go
    stale relative to the standalone benchmark repo** (§21) — this produces
    NO source diff in the `.cm` kernel itself, only in the `-D` values, so a
    plain file diff will not reveal it. A stale PTL config alone cost
    1.07-1.43x kernel-level / ~1.19x end-to-end 1st-token latency in this
    project, with zero involvement from whatever change was actually being
    reviewed at the time.

14. **When a regression is reported right after "I only changed X", verify X
    directly instead of assuming X is the cause** — twice in this project
    (§19, §21) the real cause was something adjacent that happened to ship
    in the same commit as an innocent, correctly-implemented change.

15. **To settle "is this register spill" definitively**, cross-compile the
    exact kernel + exact dispatch macros via pyopencl with
    `-mCM_printregusage` and `PYOPENCL_NO_CACHE=1` directly on the target
    device (§22) — this takes seconds and is authoritative, versus guessing
    from source inspection or trusting a cached/batched build path that may
    silently swallow the compiler's register-usage/spill report.

16. **Remote Windows (PTL) box quirks**: `rptl.py raw` does not `cd`/activate
    the venv (use full paths or `run`); `cmd.exe` needs `cd /d` to switch
    drives; prefer uploading a small Python script over fighting
    `findstr`/PowerShell quoting for anything beyond a one-liner (§22).
