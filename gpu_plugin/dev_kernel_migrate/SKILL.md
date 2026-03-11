---
name: dev_kernel_migrate
description: Migrate gpu plugin kernel to ocl_v2.
---

When migrate traditional opencl kernel selector to new ocl_v2 in gpu plugin, always include:

1. **Read GPU plugin traditional kernel selector mechanism and new ocl_v2 framework**: Understand how the GPU plugin selects kernels and how the ocl_v2 framework operates to ensure a smooth migration process. This includes familiarizing yourself with the kernel selection criteria, the structure of the ocl_v2 framework, and how it interfaces with the underlying hardware.
     - traditional kernel selector mechanism: src/plugins/intel_gpu/src/kernel_selector
     - ocl_v2 framework: src/plugins/intel_gpu/src/graph/impls/ocl_v2
     - Primitive flow: src/plugins/intel_gpu/src/graph/primitive_inst.cpp

2. **Understand How create primitive with selector opencl kernel and ocl_v2**: Learn how to create primitives using both the traditional OpenCL kernel selector and the new ocl_v2 framework. This involves understanding the API calls, data structures, and execution flow for each approach.

3. **Identify the differences between traditional kernel selector and ocl_v2**: Recognize the key differences in how kernels are selected and executed in the traditional mechanism versus the ocl_v2 framework. This includes changes in API calls, data structures, and execution flow.

4. **Migrate the kernel selection logic**: Adapt the existing kernel selection logic to fit within the ocl_v2 framework. This may involve refactoring code, updating API calls, and ensuring compatibility with the new framework's requirements.
   - If there are multiple kernels for the same primitive, create a directory for the primitive in `ocl_v2/` and implement kernel generators and impls there. The `.cl` files stay in `kernel_selector/cl_kernels/` and are referenced via `FC_EXTRA_KERNELS` in CMakeLists.txt (see Step 0).
   - If there is only one kernel for the primitive, you can directly create the kernel in ocl_v2 without a subdirectory.

Keep explanations conversational. For complex concepts, use multiple analogies.

---

## Case Study: Fully Connected (FC) Migration ✅ COMPLETED

FC is a representative complex migration: it has 18 kernel implementations in the traditional kernel_selector with a 7-nested-loop autotuning system, multiple runtime paths (DEFAULT/SLM/DynQuant), and special handling for weights reorder and ND→2D shape canonicalization. The migration to ocl_v2 is complete on branch `river/migrate_fc_to_oclv2`. Use this as the reference template for any multi-kernel primitive migration.

### Source Analysis (Traditional Framework)

Key files to read and understand before writing any ocl_v2 code:

```
src/plugins/intel_gpu/src/kernel_selector/kernels/fully_connected/
  fully_connected_kernel_selector.cpp          # registers 18 kernels, priority order
  fully_connected_kernel_bf_tiled.h/.cpp       # main path: 7-nested-loop autotuning, DEFAULT+SLM multi-kernel
  fully_connected_kernel_gemv.h/.cpp           # batch==1 fast path (INT4, compressed)
  fully_connected_kernel_bfyx_ref.h/.cpp       # reference fallback (all dtypes, all layouts)
  fully_connected_params.h                     # FC-specific params: compressed_weights, dynamic_quantization

src/plugins/intel_gpu/src/graph/impls/ocl/fully_connected.cpp   # legacy ocl impl
  # Key methods to port:
  #   update_impl_params() -> fc_canonicalize_shapes() in fc_base.cpp
  #   get_arguments()      -> get_arguments_desc() in FCKernelGenerator (fc_base.cpp)

src/plugins/intel_gpu/src/kernel_selector/auto_tuner.h/.cpp
  # TuningCache: offline cache.json format, LoadKernelOffline, computeUnits+hash key
  # bf_tiled generates 50+ tile combinations (tile_b x tile_ofm x tile_ifm x tile_k x simd x ...)
```

Also read `primitive_inst.cpp` around `ImplementationsFactory::get_primitive_impl_for_params()` (the 6-step cache):
1. Static impl cache lookup — exact-shape hit returns immediately
2. Async compilation task check
3. Dynamic impl cache search — shape-compatible hit
4. New dynamic impl creation via `create_impl()`
5. Dynamic impl update/specialization
6. Force static compile

---

### Target File Structure (As Built)

```
src/plugins/intel_gpu/src/graph/impls/ocl_v2/
  # No .cl files are copied here for FC.
  # Instead, CMakeLists.txt uses FC_EXTRA_KERNELS to reference the originals directly.
  # (see Step 0)

  fc/                                          # cpp/hpp only — no .cl files in this directory
    fc_base.hpp/.cpp    # FCKernelGenerator + FCImplBase + FCBase_IM + shape helpers
    fc_bf_tiled.hpp/.cpp  # FCBfTiled(Default/SLM/DynQuant)Generator + FCBfTiledImpl + FCBfTiled
    fc_ref.hpp/.cpp     # FCRefGenerator + FCRefImpl + FCRef
    fc_gemv.hpp/.cpp    # FCGEMVGenerator + FCGEMVImpl + FCGEMV

src/plugins/intel_gpu/src/graph/registry/
  fully_connected_impls.cpp  # registration order: onednn → FCGEMV → FCBfTiled → FCRef → legacy
```

---

### Class Hierarchy (As Built)

```
KernelGenerator (ocl_v2/utils/kernel_generator.hpp)
  └── FCKernelGenerator (fc/fc_base.hpp)        # shared JIT + arguments + canonicalization
        ├── FCRefGenerator (fc/fc_ref.hpp)       — "fully_connected_gpu_bfyx_ref"
        ├── FCBfTiledDefaultGenerator (fc/fc_bf_tiled.hpp)  — "fully_connected_gpu_bf_tiled" / "default"
        ├── FCBfTiledSLMGenerator     (fc/fc_bf_tiled.hpp)  — "fully_connected_gpu_bf_tiled" / "slm"
        ├── FCBfTiledDynQuantGenerator(fc/fc_bf_tiled.hpp)  — "fully_connected_gpu_bf_tiled" / "dyn_quant"
        └── FCGEMVGenerator (fc/fc_gemv.hpp)     — "fully_connected_gpu_gemv"

PrimitiveImplOCL (ocl_v2/primitive_ocl_base.hpp)
  └── FCImplBase (fc/fc_base.hpp)
        ├── FCRefImpl     (fc/fc_ref.hpp)       — single stage: stage_ref
        ├── FCBfTiledImpl (fc/fc_bf_tiled.hpp)  — up to 3 stages: [dyn_quant,] default, [slm]
        └── FCGEMVImpl    (fc/fc_gemv.hpp)      — single stage: stage_gemv

ImplementationManager (registry/implementation_manager.hpp)
  └── FCBase_IM (fc/fc_base.hpp)                # shared query_formats (bfyx for act + output)
        ├── FCRef     : FCBase_IM  OV_GPU_PRIMITIVE_IMPL("ocl::fc::ref")
        ├── FCBfTiled : FCBase_IM  OV_GPU_PRIMITIVE_IMPL("ocl::fc::bf_tiled")
        └── FCGEMV    : FCBase_IM  OV_GPU_PRIMITIVE_IMPL("ocl::fc::gemv")
```

---

### Step 0: Embed .cl Kernels Without Copying (CMakeLists.txt)

The `.cl` source files **stay in `kernel_selector/cl_kernels/`** and are not copied into `ocl_v2/`. Instead, update `ocl_v2/CMakeLists.txt` to pass them to the codegen script via `--extra_kernels`:

```cmake
# In src/plugins/intel_gpu/src/graph/impls/ocl_v2/CMakeLists.txt
set(KS_CL_KERNELS_DIR "${MAIN_DIR}/src/kernel_selector/cl_kernels")
set(FC_EXTRA_KERNELS
    "${KS_CL_KERNELS_DIR}/fully_connected_gpu_bf_tiled.cl"
    "${KS_CL_KERNELS_DIR}/fully_connected_gpu_gemv.cl"
    "${KS_CL_KERNELS_DIR}/fully_connected_gpu_bfyx_ref.cl"
)

add_custom_command(OUTPUT "${CODEGEN_CACHE_DIR}/${KERNEL_SOURCES}"
  ...
  COMMAND "${Python3_EXECUTABLE}" "${CODEGEN_SCRIPT}" ...
                                  -extra_kernels ${FC_EXTRA_KERNELS}
  DEPENDS ${KERNELS} ${FC_EXTRA_KERNELS} "${CODEGEN_SCRIPT}"
  ...
)
```

Also update the codegen script `common_utils/kernels_db_gen.py` to accept and process `--extra_kernels` (a list of absolute paths to `.cl` files outside the root dir).

**Critical**: The `GLOB_RECURSE *.cl` in CMakeLists.txt only scans the `ocl_v2/` source directory. Any `.cl` from another location must be explicitly listed as above and wired into the DEPENDS of the custom command.

---

### Step 1: FCKernelGenerator + FCImplBase + FCBase_IM (fc_base.hpp/.cpp)

Three distinct base classes, each serving a different role:

**`FCKernelGenerator : public KernelGenerator`** — shared JIT and argument logic for all FC stages:

```cpp
class FCKernelGenerator : public KernelGenerator {
public:
    explicit FCKernelGenerator(std::string_view name, std::string_view suffix = "")
        : KernelGenerator(name, suffix) {}
protected:
    [[nodiscard]] static kernel_impl_params get_canonical_params(const kernel_impl_params& p);
    [[nodiscard]] JitConstants get_jit_constants(const kernel_impl_params& params) const override;
    [[nodiscard]] Arguments    get_arguments_desc(const RuntimeParams& params) const override;
};
```

`get_jit_constants()` emits: INPUT0..N (activation + compressed scale/zp inputs), FILTER (weights with layout-aware macros), OUTPUT, BIAS_TERM, COMPRESSED_WEIGHTS, and full decompression scale/zp group macros.

`get_canonical_params()` calls `fc_canonicalize_shapes()` — the ND→2D reshape that all FC kernels require. This replaces `update_impl_params()` from the legacy impl.

**`FCImplBase : public PrimitiveImplOCL`** — shared lifecycle:

```cpp
struct FCImplBase : public PrimitiveImplOCL {
    explicit FCImplBase(const std::string& name) : PrimitiveImplOCL(name) {}
    void update(cldnn::primitive_inst& inst, const RuntimeParams& params) override;
    // update() calls inst.update_shape_info_tensor(params) for dynamic-shape handling
};
```

**`FCBase_IM : public ImplementationManager`** — shared format query:

```cpp
struct FCBase_IM : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("ocl::fc_base")
    explicit FCBase_IM(shape_types shape_type, ValidateFunc vf = nullptr)
        : ImplementationManager(impl_types::ocl, shape_type, std::move(vf)) {}

    [[nodiscard]] in_out_fmts_t query_formats(const program_node& node) const override {
        // Activation (input 0) → bfyx,  Output → bfyx
        // All other inputs (weights, bias, scale, zp) → format::any
    }
};
```

---

### Step 2: `get_arguments_desc` — Actual Kernel Argument Order

The **actual** argument order for FC kernels (important: differs from the legacy impl's ordering) is:

```
OPTIONAL_SHAPE_INFO (dynamic only)
INPUT[0]   — activation
INPUT[1]   — decompression_scale  (if compressed)
INPUT[2]   — decompression_zp     (if compressed and tensor ZP, not scalar)
OUTPUT[0]
WEIGHTS[0]
BIAS[0]    (if has_bias)
INTERNAL_BUFFER 0, 1  (for dynamic-quantize pre-pass, added by DynQuantGenerator)
```

```cpp
Arguments FCKernelGenerator::get_arguments_desc(const RuntimeParams& params) const {
    const auto& desc = params.typed_desc<fully_connected>();
    Arguments args;
    if (params.is_dynamic())
        args.push_back({ArgumentDescriptor::Types::SHAPE_INFO, 0});
    args.push_back({ArgumentDescriptor::Types::INPUT, 0});         // activation
    if (desc->decompression_scale.is_valid())
        args.push_back({ArgumentDescriptor::Types::INPUT, 1});     // scale
    if (desc->decompression_zero_point.is_valid() &&
        !desc->decompression_zero_point_scalar.has_value())
        args.push_back({ArgumentDescriptor::Types::INPUT, 2});     // zp tensor
    args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});
    args.push_back({ArgumentDescriptor::Types::WEIGHTS, 0});
    if (desc->bias.is_valid())
        args.push_back({ArgumentDescriptor::Types::BIAS, 0});
    return args;
}
```

**⚠️ Scale/ZP come before OUTPUT/WEIGHTS** — this matches the `.cl` kernel parameter order. Verify against the actual kernel signature whenever porting a new kernel.

---

### Step 3: DispatchDataFunc — Runtime GWS/LWS Callback

Each generator overrides `get_dispatch_data_func()` to return a lambda that sets `kd.params.workGroups` at runtime (after shapes are known):

```cpp
DispatchDataFunc FCRefGenerator::get_dispatch_data_func() const {
    return DispatchDataFunc{[](const RuntimeParams& p, KernelData& kd, ImplRuntimeParams*) {
        const auto canonical = fc_canonicalize_shapes(p);
        const auto& out = canonical.output_layouts[0];
        if (!out.is_static()) return;  // skip if shapes dynamic (handled by update())

        const bool is_3d = (out.format == format::bfyx);
        auto ps = out.get_partial_shape();
        if (is_3d && ps.size() >= 3) {
            kd.params.workGroups.global = {(size_t)ps[1].get_length(),  // OFM
                                           (size_t)ps[2].get_length(),  // Y
                                           (size_t)ps[0].get_length()}; // B
        } else {
            kd.params.workGroups.global = {(size_t)ps[1].get_length(),  // OFM
                                           (size_t)ps[0].get_length(),  // B
                                           1};
        }
        kd.params.workGroups.local = {1, 1, 1};
    }};
}
```

See `fc_bf_tiled.cpp` for the bf_tiled GWS computation (uses `round_up_to` / `ceil_div` with tile params).

---

### Step 4: FCBfTiledImpl — Multi-Stage DEFAULT/SLM/DynQuant

The bf_tiled kernel supports up to 3 execution stages selected at construction time:

```cpp
class FCBfTiledImpl : public FCImplBase {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::ocl::FCBfTiledImpl)

    // Stages declared as members — make_deep_copy() recreates them by member-order default ctor
    Stage::Ptr stage_default   = make_stage<FCBfTiledDefaultGenerator>();
    Stage::Ptr stage_slm       = make_stage<FCBfTiledSLMGenerator>();
    Stage::Ptr stage_dyn_quant = make_stage<FCBfTiledDynQuantGenerator>();

    FCBfTiledImpl() : FCImplBase(FCBfTiled_type_info_name) {}
    explicit FCBfTiledImpl(const RuntimeParams& params);  // calls add_stage() based on ctx
    ...
    // get_stages_execution_order() — runtime DEFAULT↔SLM selection by actual batch
    [[nodiscard]] std::vector<size_t> get_stages_execution_order(
        const cldnn::primitive_inst& instance) const override;
};
```

Stage ordering in the constructor:
```
slot 0: stage_dyn_quant  (if fc_bft_should_dynamic_quantize(ctx))
slot N: stage_default    (always)
slot N+1: stage_slm      (if SLM feasible AND shape_agnostic)
```

Runtime dispatch via `get_stages_execution_order()`: checks actual batch from `instance.get_impl_params()`, returns only the applicable slot indices (DQ + DEFAULT, or DQ + SLM).

**Tile selection**: Port `GetAutoTuneParams(..., KernelType::ANY, -1)` heuristic as `fc_bft_select_tune_params()`. This is the strategy-A heuristic-only approach — offline `cache.json` can be added later.

---

### Step 5: Weights Reorder — Manual `_weights_reorder_params`

In the FC ocl_v2 migration, weights reorder is handled **manually in `create_impl()`** rather than via `get_preferred_weights_layout()`. The pattern in all three FC impls:

```cpp
std::unique_ptr<primitive_impl> FCBfTiled::create_impl(
    const program_node&, const RuntimeParams& params) const
{
    // 1. Decide target weight format
    const cldnn::format target_fmt = select_weights_format(params, tp_def);
    const cldnn::format src_fmt    = params.input_layouts[1].format;

    // 2. Patch params so JIT generates FILTER macros for the post-reorder layout
    RuntimeParams patched = params;
    patched.input_layouts[1].format = target_fmt;
    if (patched.weights_layout.has_value())
        patched.weights_layout->format = target_fmt;

    // 3. Construct impl with patched params
    auto impl = std::make_unique<FCBfTiledImpl>(patched);

    // 4. Set weights reorder if format actually changes
    if (src_fmt != target_fmt) {
        layout src_l = params.input_layouts[1];
        layout dst_l = src_l;
        dst_l.format = target_fmt;
        impl->set_weights_reorder_params(
            std::make_shared<WeightsReorderParams>(src_l, dst_l));
    }
    return impl;
}
```

Each impl exposes `set_weights_reorder_params()` as a public method because `_weights_reorder_params` is `protected` in `primitive_impl`:

```cpp
void set_weights_reorder_params(std::shared_ptr<cldnn::WeightsReorderParams> wrp) {
    _weights_reorder_params = std::move(wrp);
}
```

---

### Step 6: FCRef — Single-Stage Fallback

`FCRef` is the simplest impl: one stage, one kernel, `oiyx` weight layout, accepts all dtypes and layouts (rejects only 4D outputs):

```cpp
// get_jit_constants emits:
//   OUTPUT_3D      — if output format == bfyx
//   ACCUMULATOR    — f32 for int weights, i32 for int8 input, else input dtype
//   ACTIVATION     — f32 for int8 input, else input dtype
//   _TYPED activation macros
//   INT4_PACKED_TYPE — if weights are i4/u4 (pack_size=2)
//   fused ops      — idx_order {"b","ofm","0","0"} or {"b","ofm","oym","0"} for 3D

// validate_impl rejects:
//   bfyx output with X.v > 1 (4D output)
//   input dtype not in {f16, f32, i8, u8}
```

---

### Step 7: FCGEMV — Batch==1 INT4 Fast Path

Key validation requirements (all must pass):
- `decompression_scale.is_valid()` (compressed weights)
- weight dtype == i4 or u4
- input dtype == f16
- output dtype == f16 or f32
- batch == 1 (static) or dynamic with appropriate weight layout
- IFM % 16 == 0, scale_group_size % 16 == 0
- no swiglu fused op
- no bfyx input padding (`static_cast<bool>(in0.data_padding)` is false)
- no 4D output

Weight layout selection (`select_weights_format`):
```
os_is_yx_osv64_isv2  — if OFM >> IFM (horizontal, large-N)
os_iyx_osv16         — default (also covers vertical / large-K case)
(os_is_yx_osv32_isv2 — only if weight is already in this layout)
```

JIT constants specific to GEMV:
```
FILTER_LAYOUT_OS_IS_YX_TYPE  — 0 (osv16) / 1 (osv32_isv2) / 2 (osv64_isv2)
WEI_UINT4                    — 1 if UINT4, 0 if INT4
SIMD = 16
WEIGHTS_K = IFM.v
WEIGHTS_N = OFM.v
ACTIVATION type always F32 (even for F16 input)
fused-ops idx_order: {"0","0","(cur_n + i)","0"} for osv16
                     {"0","0","(cur_n + 16*i)","0"} for osv32/64
```

GWS: `{OFM, 1, 16}` for osv16 / `{OFM/2, 1, 16}` for osv32 / `{OFM/4, 1, 16}` for osv64 — LWS: `{16, 1, 16}`

---

### Step 8: Registration (fully_connected_impls.cpp)

Final registration order (highest priority → lowest):

```cpp
#include "impls/ocl_v2/fc/fc_bf_tiled.hpp"
#include "impls/ocl_v2/fc/fc_gemv.hpp"
#include "impls/ocl_v2/fc/fc_ref.hpp"

static const std::vector<std::shared_ptr<ImplementationManager>> impls = {
    OV_GPU_CREATE_INSTANCE_ONEDNN(onednn::FullyConnectedImplementationManager, shape_types::static_shape)
    OV_GPU_CREATE_INSTANCE_OCL(ocl::FCGEMV, shape_types::static_shape)
    OV_GPU_CREATE_INSTANCE_OCL(ocl::FCGEMV, shape_types::dynamic_shape,
        [](const program_node& node) {
            if (node.can_use(impl_types::onednn)) return false;
            return node.get_output_pshape().size() <= 3;
        })
    OV_GPU_CREATE_INSTANCE_OCL(ocl::FCBfTiled, shape_types::static_shape)
    OV_GPU_CREATE_INSTANCE_OCL(ocl::FCBfTiled, shape_types::dynamic_shape,
        [](const program_node& node) {
            if (node.can_use(impl_types::onednn)) return false;
            return node.get_output_pshape().size() <= 3;
        })
    OV_GPU_CREATE_INSTANCE_OCL(ocl::FCRef, shape_types::static_shape)
    OV_GPU_CREATE_INSTANCE_OCL(ocl::FCRef, shape_types::dynamic_shape,
        [](const program_node& node) {
            if (node.can_use(impl_types::onednn)) return false;
            return node.get_output_pshape().size() <= 3;
        })
    // Legacy OCL fallback kept during transition
    OV_GPU_GET_INSTANCE_OCL(fully_connected, shape_types::static_shape)
    OV_GPU_GET_INSTANCE_OCL(fully_connected, shape_types::dynamic_shape,
        [](const program_node& node) {
            if (node.can_use(impl_types::onednn)) return false;
            return node.get_output_pshape().size() <= 3;
        })
};
```

Also add `BIND_BINARY_BUFFER_WITH_TYPE` at the bottom of **each** impl `.cpp` file for serialization:
```cpp
// At bottom of fc_bf_tiled.cpp:
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::ocl::FCBfTiledImpl)

// At bottom of fc_ref.cpp:
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::ocl::FCRefImpl)

// At bottom of fc_gemv.cpp:
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::ocl::FCGEMVImpl)
```

---

### Key Constraints Checklist

| Constraint | Where it bites |
|---|---|
| `.cl` files for external kernels need `FC_EXTRA_KERNELS` | CMakeLists.txt `GLOB_RECURSE *.cl` only at ocl_v2 root; files in other directories must be listed explicitly |
| `kernels_db_gen.py` must accept `--extra_kernels` | The codegen script must be updated to process extra `.cl` files from outside the root |
| `fc_canonicalize_shapes()` required before any JIT/dispatch | All FC kernels expect 2D `[B, K]` input; ND activations must be reshaped first |
| Patch `patched.input_layouts[1].format` before constructing impl | The JIT generator reads the weight layout from `params.input_layouts[1].format`; must reflect the target (post-reorder) format |
| `SHAPE_INFO` must be first arg in dynamic path | `make_base_jit_constants` emits `OPTIONAL_SHAPE_INFO_ARG` only when `IS_DYNAMIC` defined |
| `DECLARE_OBJECT_TYPE_SERIALIZATION` + `BIND_BINARY_BUFFER_WITH_TYPE` required | Needed for impl blob save/load from GPU kernel cache |
| Stage members must be declared in the class (not only in ctor) | `make_deep_copy()` uses the default ctor and relies on member initializer order to recreate `_stages` |
| `cldnn::format` is not literal — use `const`, not `constexpr` | `format` has a non-trivial destructor so cannot be `constexpr` |
| `cmake --build` after adding new .cpp files requires `cmake ..` first | New files added after a configure step won't be picked up by GLOB until CMake re-runs |
| `data_padding` validity check: use `static_cast<bool>(layout.data_padding)` | `padding::operator bool()` returns true if any lower/upper size is non-zero |

---

### ocl_v2 Framework Optimizations Applied

The following performance improvements were applied to the ocl_v2 framework itself during the FC migration. They benefit all ocl_v2 primitives:

| File | Change | Benefit |
|---|---|---|
| `utils/kernel_generator.cpp` | `make_tensors_jit_constants`: replaced `static std::map + std::mutex` with `thread_local std::unordered_map` | Eliminates lock contention across parallel compilation streams |
| `utils/kernels_db.cpp` | `get_kernel_template` / `get_kernel_header`: replaced O(n) linear scan with `static std::unordered_map` built once | O(1) lookup for all kernel DB queries |
| `utils/kernels_db.cpp` | Added `SvHash` + `std::equal_to<>` transparent hash/equal on the `KernelSourceMap`; `map.find(template_name)` now takes `string_view` directly | Eliminates one heap `std::string` allocation+free per kernel template / header lookup |
| `utils/kernel_generator.cpp` | `build_code`: pre-fetches template once, calls `code.reserve()` before assembly | Avoids double DB lookup and string reallocation |
| `utils/kernel_generator.cpp` | `build_code`: `code.add_line(kernel_template)` via `string_view` overload | Eliminates a >100 KB heap copy of the kernel source per compilation |
| `utils/kernel_generator.cpp` | `make_base_jit_constants`: `jit_constants.reserve(7)` before adding base constants | Avoids the default vector zero-alloc + double-growth realloc for 5/7 elements |
| `utils/kernel_generator.cpp` | `get_jit_constants`: pre-computes `tensors` size, calls `jit.reserve(jit.size() + tensors.size())` before merge | Single allocation for the full merged JIT constant vector |
| `common_utils/jitter.hpp` | Added `CodeBuilder::reserve(size_t capacity)` method | Allows callers to pre-allocate ostringstream buffer |
| `common_utils/jitter.hpp` | Added `CodeBuilder::add_line(std::string_view)` overload | Zero-copy path for writing large string_view content (kernel template) into the code builder |
| `primitive_ocl_base.hpp` | `update_stages_flags()`: iterates `_order` (active stages only) instead of all `_stages` | Avoids writing flags to inactive stages on every inference frame |
| `primitive_ocl_base.hpp` | `make_deep_copy()`: skips inactive stages (not in `_order`) | Avoids unnecessary `KernelData` copy and `kernel->clone()` GPU handle allocation |