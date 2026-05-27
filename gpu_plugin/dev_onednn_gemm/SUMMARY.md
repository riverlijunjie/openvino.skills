# oneDNN GEMM Kernel — Design and Optimization Summary

Source root:
`src/plugins/intel_gpu/thirdparty/onednn_gpu/src/gpu/intel/gemm/`

The Intel-GPU GEMM in oneDNN is a **JIT-generated, template-instantiated assembly kernel** built on top of nGEN (an assembler-level C++ DSL for Gen ISA). The generator core is internally called *gemmstone*. A pre-tuned **kernel catalog** describes hundreds of (HW × precision × layout) recipes, each encoding a "strategy" string consumed by a strategy parser that drives the code generator. There is also a newer high-level **DSL builder** (`generator_dsl/`) used optionally (`enable_generator_dsl()` env var).

---

## 1. High-level architecture

| Layer | Files | Role |
|---|---|---|
| oneDNN integration (primitive descriptor + dispatch) | `jit/pd.{cpp,hpp}`, `jit.{cpp,hpp}`, `gen_kernel.{cpp,hpp}` | Bridges OpenVINO/oneDNN primitive system to gemmstone; converts memory descriptors → `GEMMProblem`, picks an `Entry` from the catalog, calls strategy parser, JITs and launches. |
| Kernel catalog | `jit/selector/db/kernel.db` (+ `ukernel_*.db`), `jit/include/gemmstone/kernel_catalog.hpp`, `selector/kernel_selector.cpp`, `selector/kernel_evaluator.cpp` | Sorted DB of `Entry{Selector, Restrictions, strategy_string, driverInfo}`. Selector is keyed by `(hw, model, precisions[A,B,C], layouts)`. Evaluator scores entries against problem dims using a perf model (W-model) and picks best. |
| Strategy parser | `jit/generator/strategy_parser.cpp`, `strategy.cpp` | Parses the strategy string (a compact tuned recipe) into the `GEMMStrategy` struct: access types, unrolls, prefetch distances, SLM buffer count, walk order, systolic flags, etc. Runs `preflight()` to derive defaults and check legality. |
| Generator core | `jit/generator/generator.cpp` (umbrella) + `pieces/*.cxx` (template-instantiated per HW) | The actual ISA emitter. One `Generator<hw>` class is instantiated for Gen12LP, XeHP, XeHPG, XeHPC, Xe2, Xe3, Xe3p (`hw_template_instantiations.cxx`). |
| Optional DSL path | `jit/generator_dsl/`, `jit/dsl/` | Higher-level expression-IR alternative to nGEN-direct code generation. |
| Selection of micro-kernels (e.g., for SDPA fusion) | `jit/generator/microkernel_provider.cpp`, `microkernel_selector.cpp` | Embeds GEMM as a "ukernel" callable from other kernels (used for flash-attention-style fusion). |
| Reference paths | `ref.{cpp,cl,hpp}`, `with_post_ops.{cpp,cl,hpp}`, `jit_xe_hp_systolic.*`, `xe_*systolic_copy.cl` | Fallback OpenCL kernels and the legacy hand-rolled Xe-HP systolic copy/compute paths. |

### Data flow when launching one GEMM

1. `gen_desc_t::finalize()` (`gen_kernel.cpp`) ← entry chosen by `selectKernel()` from `catalog()`.
2. Strategy string → `parseStrategy()` populates `GEMMStrategy`. Hardware-aware fixups follow (e.g., `raHW = XeHPC` on Xe2/Xe3 to reuse PVC strategies; bumping A/B alignment to 16B when block-2D loads are tagged).
3. `GEMMStrategy::preflight()` derives default SIMD, alignments, atomic eligibility, named-barrier requirements, prefetch downgrades when block-2D is illegal on non-packed inputs, etc.
4. `Generator<hw>::gemm()` emits the kernel: prolog → walk-order address setup → L3 prefetch warmup / TLB warmup → main k-loop (with SLM copies and DPAS pipeline) → C update / post-ops → epilog.
5. The catalog's `driverInfo.unroll`, alignments and "tags" feed back into runtime workgroup sizing and walk-order argument plumbing (`hilbert_vd`, `bslice`, etc.).

---

## 2. Kernel design and data flow inside the k-loop

The compute structure is a classic *outer-product accumulation* of an `unrollM × unrollN` C-tile, over `unrollK` k-slice per iteration, with three pipelined data movements:

```
 Global ─(load)─►   Register (load-buffer, A_copies / B_copies deep)
       └─(L3 PF)─►  L3 cache (l3_prefetch.cxx, distance = prefetchABL3)
       └─(L2 PF)─►  L1/L2     (A_prefetch / B_prefetch, distance = prefetchA/B)
       └─(coop)──►  SLM (slmBuffers = 0/1/2/3; cooperative split among WG threads)
 SLM ─(load)─►     Register (A/B operand registers)
 A,B reg ─(DPAS / DPASW / FMA)─► C accumulator registers
 C reg ─(repack/post-op/atomic)─► Global
```

Key pieces (under `jit/generator/pieces/`):

- `gemm.cxx` – top-level kernel scaffolding (`gemmBody`, prolog, register-allocator init).
- `gemm_setup.cxx` – layout planning for A/B/C tiles, repack layouts (`Ar_layout`, `Br_layout`), crosspack, decisions like `upConvertATo8Bit` (Xe3p, int4→int8 pre-DPAS), `cRepackPeriod`/`cRepackPanel`.
- `gemm_microkernel.cxx` – inline-callable GEMM micro-kernel form.
- `k_loop_setup.cxx`, `k_loop.cxx` – emits the unrolled, pipelined k-loop with `loop_sequencer` to overlap loads, DPAS, prefetches, and SLM writes/fences.
- `monolithic_k_loop_dpasw.cxx` – the legacy fully-unrolled DPASW path used for `fixedSystolic` 32×32 / 32×48 (`slmABufBlockSize() = 1152`, `slmBBufBlockSize() = 1536`).
- `matrix_multiply.cxx` – emits the actual `dpas`/`dpasw`/`mad` instruction trees (systolic depth 8, repcount/sdepth derived from `outerProductCount()`).
- `matrix_access.cxx` + `address_setup.cxx` – emits load/store with the selected `AccessType` (Block / PseudoBlock / Scattered / ChannelScattered / Block2D / Block2DTranspose / Block2DVNNI / CacheLine).
- `c_update.cxx` – beta/alpha apply, C type-cast & repack, atomic / non-atomic writeback.
- `post_ops.cxx` – fused eltwise / binary / sum / quantization-output post-ops.
- `quantization.cxx`/`.cpp/.hpp` – 2D scale / zero-point handling for weights-only quantization.
- `copy.cxx` + `copy_plan.{cpp,hpp}` – `CopyPlan`: an IR that figures out *how* to move/repack/convert data between regions. Critical for int4/int8 → f16 upconversion and crosspack repacking.
- `l3_prefetch.cxx` – emits a separate cooperative L3-prefetch front-end (Xe2+).
- `tlb_warmup.cxx` – pre-touches A/B pages to avoid TLB-miss bubbles for large persistent kernels.
- `stream_k.cxx` – Stream-K decomposition for small-M/N or large-K problems.
- `walk_orders.cxx` – emits the WG → (m,n,k) translation per the chosen walk order.
- `register_layout.{hpp,cpp}` – `RegisterBlock` / `RegisterLayout`: source of truth for register-resident tiles, crosspack, masks, leading dim, fragmentation, message size.
- `register_allocation.cxx`, `allocators.{hpp,cpp}` – nGEN-level GRF bank-aware allocator; `raHW` is forced to XeHPC on Xe2/Xe3 so PVC strategies remain valid.
- `masks.cxx`, `remask.{cxx,cpp,hpp}` – tail/remainder handling (variable + fixed masks, descriptor-based remainders, fragmentation).
- `cooperative_split.{cpp,hpp}` – splits a tile across the workgroup for cooperative global→SLM copies and cooperative prefetches (`CoopSplit::{MN,K,Linear}`).
- `atomic_fusions.cxx` – fused atomic-add C update used with `kParallelVariable` / Stream-K reductions.
- `row_column_sums.cxx` – on-the-fly A-row / B-col sum reduction for integer GEMMs with asymmetric quantization.
- `bfn.{cpp,hpp}` – Boolean-function (bfn) instruction synthesis, used for fast int4→f16 (see §5).

---

## 3. Vectorization & instruction usage

### SIMD width
`fmaSIMD` defaults to `min(32, 2*GRF_bytes / max(Ta,Tb,Tc).size)` (`strategy.cpp` `preflight`). On Xe2/Xe3 with 64-byte GRFs and f16, that yields SIMD32. `subgroupSize` is at least `GRF_bytes/4` (i.e. 16 on XeHPG, 16 on Xe2/3 sub-group APIs).

### Systolic (`DPAS`/`DPASW`)
- Enabled when `strategy.systolic && hw ≥ XeHP`. `systolicAvailable` is set per HW.
- **DPAS** does 8×systolic-depth outer products with repcount `r` → C[r×exec_size] += A[r×depth] · B[depth×exec_size]. For f16/bf16/int8: depth=8, exec=8 (XeHP/HPG) or exec=16 (XeHPC/Xe2/Xe3).
- **DPASW** (`strategy.dpasw`) shares B operand across fused EU pair (Gen12LP/XeHP/XeHPG) — halves B-register pressure. Disabled on Xe-HPC and later (no fused EUs).
- **fixedSystolic** path emits a hand-tuned 32×32 (XeHP) or 32×48 (XeHPC) systolic inner loop with fixed SLM block sizes (1152/1536 bytes for A/B).
- **extendedAtomicFMA** allows merging the FMA into an atomic-store sequence on XeHPC+ for very small panels.

### FMA fallback
For pre-XeHP or precision combos without DPAS support (e.g. fp32 GEMM on Xe-LP), gemmstone emits classical `mad` chains via `matrix_multiply.cxx`. Crosspack and tile sizing are still chosen so register region restrictions are satisfied.

### Emulation
`emulation.cxx` provides software emulation of 64-bit ops, 32×32 multiplies, bf16 arithmetic on HW lacking it (Gen12LP), int4 upconversion, hf→bf8, bf→f32 narrowing, etc.

---

## 4. Memory access patterns & blocking

### Three-level tiling
1. **Workgroup tile** `wg[LoopM] × wg[LoopN]` threads, each producing an `unrollM × unrollN` C-tile. K is split by `wg[LoopK]` (if `kParallelLocal`).
2. **Register tile** per thread: `unrollM × unrollN` accumulators; A is loaded as `unrollM × ka_load`, B as `kb_load × unrollN`. `A_copies`/`B_copies` (1–4 deep) is the load buffer depth used to hide latency.
3. **Crosspack** – elements are physically interleaved inside GRF rows to match DPAS's required operand layout (e.g. f16 crosspack=2 makes successive k-elements adjacent within 32-bit lanes).

### Access types (per matrix and per usage: data load, prefetch, unaligned)
Selected by the strategy character: Block (HDC block messages), PseudoBlock, Scattered, ChannelScattered, **Block2D** / **Block2DTranspose** / **Block2DVNNI** (XeHPC+ 2D block-load with built-in transpose or VNNI repack), CacheLine. `RegisterBlock` decides per-instruction `ebytes`, `count`, `simdSize`, mask layout. `downgradeBlock2D()` falls back when alignment/tag conditions are violated.

### SLM staging (`slmBuffers` ∈ {0,1,2,3,4})
- 0 → no SLM, registers are loaded directly from global.
- 1 → single buffer, barrier-twice per unrollK iteration.
- ≥2 → double/triple-buffering: while one buffer is consumed by DPAS the next is being filled.
- SLM block sizes: `slmABufBlockSize = unrollM * unrollKSLM * Ta` (`strategy.hpp`).
- Cooperative split (`coopA`/`coopB`, see `cooperative_split.cpp`): each workgroup thread copies a slice of the A or B SLM tile (split by `K`, `MN`, or `Linear`), maximizing global-load coalescing.

### Prefetching (three independent layers)
1. **L1/L2 prefetch** (`prefetchA`, `prefetchB`, distance in units of `unrollK`). Uses dedicated `A_prefetch`/`B_prefetch` strategies. Block-2D prefetch is downgraded automatically when not legal (`downgradeAPFAccess`).
2. **L3 prefetch** (`l3PrefetchA`, `l3PrefetchB`, `prefetchABL3`, `ka_prefetchL3`, `kb_prefetchL3`) — Xe2+ explicit L3 hinted loads emitted by `l3_prefetch.cxx`, often cooperatively across the WG (`cooperativePF = true`).
3. **TLB warmup** (`tlbWarmup`) — `tlb_warmup.cxx`. Issues throw-away loads into A/B before the main loop to pre-populate TLB pages.

### Caching control
`getCaching()` in `strategy_parser.cpp` lets each access encode L1/L2/L3 cache hints (`L1C_L3C`, `L1UC_L3WB`, etc.). XeHPC default reads cached, writes uncached-L1/write-back-L3. Xe3p adds explicit L2 control.

### Atomics
`useAutoAtomic()` enables atomic C add on XeHPG+ when β==1, native atomic add exists for `Tc_ext`, no incompatible post-op, and C is not block-2D. Float atomics are gated on HW and address model.

### Granular masking / remainder
`RemainderOptions {AvoidFragment, AllowFragment, AllowDescriptors, NoFixedMasks, …}`. The two-mask system (`rowMask`/`colMask`, each `MaskInfo` variable or fixed) lets one message handle both row and column tails. `altCRemainder` and `block2DCRemainder` choose between a fast 2D-block C path and a scattered fallback for the C tile.

### Walk orders (cache-aware m/n schedule of workgroups)
Defined in `WalkOrder` (`strategy.hpp`):

| Token | Strategy enum | What it does | Where |
|---|---|---|---|
| `li` | `SimpleLinear` | Row-major or column-major linear sweep over WG grid. | `walk_orders.cxx::gemmSimpleLinearOrder` |
| `nl` | `NestedLinear` | Two-level linear: outer panel, inner linear within panel; reuses A or B panel in L2/L3. | same |
| `hi` | `Hilbertlike` | Cache-oblivious Hilbert curve over the WG grid. Recursive bisection bails out at `hilbertBail`. | `gemmHilbertlikeOrder` |
| `bo` | `Boustrophedon` | Cache-aware panel boustrophedon (zig-zag) — keeps A or B panel hot. | `gemmBoustrophedonOrder` |
| (default) | `HW2D` | Use the GPU's native 2D dispatch ordering. | `gen_kernel.cpp:945` |

Hilbert/Boustrophedon also pass tuning constants through the kernel ABI (`hilbert_vd`, `hilbert_uvd_recip`, `hilbert_bail`, `bslice`, `bthresh`). `gen_kernel.cpp::finalize` falls back to `SimpleLinear` when scrambling/TLB warmup/3D dispatch are in play.

### Stream-K
`stream_k.cxx` decomposes work along the K dimension across persistent WGs, with atomic-add or staged-temp reduction (`fuseBeta`, `kParallelVariable`, `kPadding`). Used when M·N work is small relative to chip occupancy.

---

## 5. f16 weight GEMM optimization (and weights-only int4/int8 with f16 compute)

f16 is a first-class precision: it enters the catalog as the `f16` selector char and benefits from *all* generic strategies above. On top of that, several optimizations are f16-specific or f16-decisive:

### 5.1 Native DPAS support
- DPAS f16×f16→f32 is native on XeHP+. f16 is the "natural" operand for systolic GEMM after bf16.
- Operand layout: f16 with `crosspack=2` (two consecutive k-elements packed into a 32-bit channel). The `RegisterLayout` and copy planner ensure this crosspack on load or on repack from SLM.
- Atomic f16/bf16 C add is only native from **Xe3p** onward (`hasNativeAtomicAdd()` in `hw_utils.hpp`). On earlier HW, f16 output goes through a non-atomic C update and an optional accumulator-precision upcast (Tc=f32 accumulator).

### 5.2 2D block loads for f16
- `AccessType::Block2D` (and `Block2DVNNI` for B) are emitted on XeHPC/Xe2/Xe3 when alignment ≥ 16B (forced up in `gen_kernel.cpp::finalize`). 2D-block-VNNI delivers B already in the crosspacked form DPAS wants, eliminating a software repack.
- Block-2D transpose handles row-major B (or column-major A) without a software transpose.

### 5.3 Accumulator vs output dtype
- `Tc` (compute) is typically f32 even when `Tc_ext` is f16 (`problem.Tc.bits() != Tc_ext.bits()` triggers a fused-post-op C-cast). `c_update.cxx` emits a cast at the end of the k-loop; this is overlapped with the next tile's loads via `cLoadAhead`.
- `cRepackPanel` / `cRepackPeriod` choose how often the f32→f16 reduction is materialized, balancing register footprint vs ILP. For 2D quantized inputs the period aligns with `aqGroupK`/`bqGroupK`.

### 5.4 Weights-only quantization with f16 activations (int4 / int8 weight, f16 act / f16 compute)
This is the optimized path for LLMs (Q4_K, AWQ, GPTQ-style). Implemented mainly in `quantization.cxx`, with the *copy planner* (`copy_plan.cpp`) handling the bit-twiddling:

- **2D scale + zero-point grouping**: `aqGroupM/K` and `bqGroupN/K` parameterize a group along (M or N) and K (e.g. group-size 32 or 128). `aScale2D()`/`bScale2D()` → emit per-group dequantize fused inside the k-loop.
- **Late vs early scale / offset** (`lateScale`, `lateOffset`):
  - *Early* dequantize converts int weights to f16/f32 before DPAS — needed when the operand is too narrow to feed DPAS directly.
  - *Late* dequantize applies scales *after* a sub-accumulation, so the actual DPAS runs on the integer operand (int8 path on Xe3p). Selected when `Txs.size > Tx.size`, when `useBDPAS` is set, or when `forceLateQuant()` so demands. Reduces dequant ops by a factor of K-group.
- **"int4 special path"** (`int4SpecialPath = Tx_ext.isInt4() && Tx ∈ {f16,f32}`): uses a clever **bias-trick f16 reinterpretation** to dequant int4→f16 in 2 ops:
  1. `bfn` (`ctrl=0x6A`, src0 ^ (src1 & src2)) packs 4 low bits of int4 into the mantissa of an f16 value with exponent bits set to a known bias (`0x6400` for u4, `0x6400|8`=0x6408 for s4 with +8 shift). The result is the int value as an f16 plus a power-of-2 offset.
  2. `add` (or `mad` for s4) by `Immediate::hf(bias|0x8000)` subtracts the bias, leaving the true integer-as-f16. Then a final `mul/mad` by `0x0C00` (= 2⁻¹²) rescales — meaning subsequent group-scale multiplication can absorb the 2¹² factor; offsets are stored pre-scaled by 2⁻¹².
  (See `copy_plan.cpp::planInt4ToF16` and `quantization.cxx::gemmRepack2DOffsetData` for the exact sequence.)
- **Generic int4 upconversion** (`copy_plan.cpp::planInt4Upconversion`): when targeting non-f16 destinations or when the bias-trick is not applicable, int4 elements are byte-extracted with `and/shr` (or a Xe3p-specific `shfl` upconversion `planShflUpconvertXe3p`).
- **Xe3p s4/u4 → s8/u8 upconvert** before DPAS (`gemm_setup.cxx::upConvertATo8Bit/upConvertBTo8Bit`): on Xe3p the systolic engine has native int8 DPAS but no int4 DPAS, so weights are widened to 8-bit during the SLM-to-register repack. `crosspack` is halved (`crosspackA/2`) so the byte-wide operand still fits the same DPAS tile.
- **Lazy repack** (`gemm_setup.cxx`): for int4→f16/bf16/f32 conversions, the repacked register footprint is reduced (`ka_repack = min(ka_load, kb_load)`) to relieve GRF pressure and improve load pipelining.
- **Group-aligned k-slice** (`gen_kernel.cpp::finalize`): when `kParallelLocal && quantized2D`, `k0` is rounded up to `aqGroupK`/`bqGroupK` so each thread handles whole groups, eliminating cross-thread dequant accumulation issues.
- **Asymmetric quantization → row/column sums**: `row_column_sums.cxx` accumulates Σ A or Σ B in registers during the k-loop and subtracts `zeroPoint·Σ` at C-update. Avoids a separate reduction kernel.
- **Pad-region duplication** (`gemmRepack2DQuantizationData`): when crosspack > 1, the same dequantized value is broadcast across the padded crosspack lanes so it can multiply correctly with each k-strand.
- **bf16-special path**: for bf16 activations with f32 scales and MN-group > 1, an 8-element `qPairs` register chunk is allocated to broadcast the scale across the 2-element crosspack (`bfSpecialPath` in `quantization.cxx`).
- **f8_e8m0 scales on Xe3p**: when scales themselves are MX-style microscaling (e8m0), `Txs_int = f8_e8m0` is kept through the path to use HW-supported scaling rather than upcasting.

### 5.5 Late C cast / mixed-precision
- For f16 GEMM with f32 accumulation and α≠1 / β≠0, post-op fusion (`fusePostOps`) materializes α·acc + β·C in f32 and casts at the very end. `relaxedAccumulation` lets HW with native f32-atomic-add use the accumulator path while keeping output in f16.

### 5.6 Microkernel reuse (flash-attention-style fusion)
- For the SDPA fused kernel, `microkernel_provider.cpp` selects an f16 GEMM micro-kernel from `selector/db/ukernel_*.db` and emits it inline. This lets attention kernels reuse exactly the same DPAS pipeline as standalone GEMM, including 2D-block loads and crosspack handling.

---

## 6. Threading & parallelization

- **Workgroup layout**: `wg[LoopM] × wg[LoopN] × wg[LoopK]` HW threads. Each thread has `subgroupSize` SIMD lanes. With 64-byte GRF and `GRFs=256`, `threadsPerEU` is 4 (PVC-style); else 8.
- **Named barriers** (`namedBarriers[LoopM]`, `namedBarriers[LoopN]`): row-partial / column-partial sync on XeHPC+, used so DPAS-issuing threads don't all wait on a single global barrier per K-slice.
- **`kParallelLocal`**: split K within a workgroup, with SLM-based partial reduction.
- **`kParallelVariable`**: per-WG variable K range (Stream-K).
- **Persistent threads** (`persistent`): one launched WG sweeps multiple tiles via a global tile counter (`gemmReorderGlobalIDs`), amortizing prolog cost and enabling tile-to-tile L1/L2 reuse via walk order.
- **Fuse beta / fuse post-ops** (`fuseBeta`, `fusePostOps`): combine atomic C reduction with β-scaling and post-ops in a single epilog, gated on C atomics availability.
- **Barriers vs slmFenceWARWA**: WAR/WAW workaround fences emitted on XeHPG+ before reusing SLM buffers.

---

## 7. Optimization strategies — consolidated checklist

Cross-cutting tactics applied (and how each is enabled in the strategy string / catalog):

| # | Strategy | Knob(s) | Where emitted |
|---|---|---|---|
| 1 | DPAS / DPASW systolic compute | `systolic`, `dpasw`, `fixedSystolic` | `matrix_multiply.cxx`, `monolithic_k_loop_dpasw.cxx` |
| 2 | 2D block loads (with VNNI / transpose) | access type `m`/`v`/`t` | `address_setup.cxx`, `matrix_access.cxx` |
| 3 | Crosspack-aware register layout | `crosspack` in `RegisterBlock` | `register_layout.{cpp,hpp}` |
| 4 | Multi-deep load buffers (load pipelining) | `A_copies`, `B_copies` (`x2`,`x3`,`x4`) | `k_loop.cxx`, `gemm_setup.cxx` |
| 5 | SLM single/double/triple buffering | `slmBuffers`, `slmA`, `slmB` | `copy.cxx`, `k_loop.cxx` |
| 6 | Cooperative global→SLM copy | `coopA`/`coopB`, `cooperativePF` | `cooperative_split.cpp`, `copy.cxx` |
| 7 | L1/L2 prefetch chain | `prefetchA`, `prefetchB`, `prefetchAMasked` | `k_loop.cxx`, `matrix_access.cxx` |
| 8 | Independent L3 prefetch | `l3PrefetchA/B`, `prefetchABL3`, `ka_prefetchL3` | `l3_prefetch.cxx` |
| 9 | TLB warmup | `tlbWarmup` | `tlb_warmup.cxx` |
| 10 | Cache hints (LSC) | `{l1l3}` / `{l1l2l3}` syntax | `strategy_parser.cpp`, `matrix_access.cxx` |
| 11 | Cache-aware walk orders | `cWalkOrder` (`li`,`nl`,`hi`,`bo`) | `walk_orders.cxx` |
| 12 | Persistent threads | `persistent` | walk-order tile counter |
| 13 | Stream-K K-decomposition | `kParallelVariable`, `kPadding`, `fuseBeta` | `stream_k.cxx`, `atomic_fusions.cxx` |
| 14 | K-local reduction | `kParallelLocal`, `wg[LoopK]` | `k_loop.cxx`, SLM reduction |
| 15 | Named barriers per row/col | `namedBarriers[LoopM/N]` | barrier emission in `k_loop.cxx` |
| 16 | Atomic C add (β=1) | `autoatomic`, `C.atomic` | `c_update.cxx`, `atomic_fusions.cxx` |
| 17 | Fused post-ops + α/β + dtype cast | `fusePostOps`, `cRepackPanel`, `relaxedAccumulation` | `post_ops.cxx`, `c_update.cxx` |
| 18 | Variable + fixed mask remainders | `remHandling`, `altCRemainder`, `block2DCRemainder` | `masks.cxx`, `remask.cxx` |
| 19 | GRF bank-aware register allocation | `raHW`, `GRFs` | `register_allocation.cxx`, allocators |
| 20 | 64-bit emulation / bf16/hf emulation | `emulate64`, `emulateDWxDW` | `emulation.cxx` |
| 21 | Weights-only 2D dequant (scales+zp) | `aqGroup{M,K}`, `bqGroup{N,K}`, `aScale2D`, `bScale2D` | `quantization.cxx`, `quantization.cpp` |
| 22 | Late vs early scale/offset | `lateScale2D{A,B}`, `forceLateQuant`, `useBDPAS` | `quantization.cxx` |
| 23 | int4 → f16 bias-trick (`bfn`+`add`) | `int4SpecialPath` | `copy_plan.cpp::planInt4ToF16`, `quantization.cxx::gemmRepack2DOffsetData` |
| 24 | int4 → int8 widening for DPAS (Xe3p) | `upConvertATo8Bit`, `upConvertBTo8Bit` | `gemm_setup.cxx` |
| 25 | Lazy / partial repack | `ka_repack`, `repackA/B` | `gemm_setup.cxx`, `copy.cxx` |
| 26 | Crosspack padding broadcast | duplicate-in-pad | `quantization.cxx::gemmRepack2DQuantizationData` |
| 27 | On-the-fly A/B sums for asym-quant | `needsASums`, `needsBSums` | `row_column_sums.cxx` |
| 28 | Catalog + scoring (perf model) | `kernel_evaluator.cpp`, `kernel.db` | runtime kernel pick |
| 29 | Optional inlined micro-kernel | `ukernel_*.db`, `microkernel_provider.cpp` | SDPA / fusion sites |
| 30 | Strategy fixups for Xe2/Xe3 reusing PVC entries | `raHW = XeHPC`, align bump for block-2D | `gen_kernel.cpp::finalize` |

---

## 8. File map (quick reference)

OpenVINO-side oneDNN GPU GEMM tree:
- `jit.{cpp,hpp}` – Primitive descriptor and dispatch entry point.
- `gen_kernel.{cpp,hpp}` – Catalog lookup, strategy parse, kernel JIT call.
- `gen_kernel_db.{cpp,hpp}` – Provides `catalog()` (the compiled kernel DB).
- `pd.{cpp,hpp}` – oneDNN primitive descriptor with attr/post-op handling.
- `with_post_ops.{cpp,hpp,cl}` – Reference + OpenCL GEMM-with-post-ops.
- `ref.{cpp,hpp,cl}` – Reference fallback.
- `jit_xe_hp_systolic.{cpp,hpp}`, `xe_hp_systolic_copy.cl`, `xe_hpc_systolic_copy.cl`, `xe_systolic_copy_kernel.hpp` – Legacy hand-written systolic copy/compute used on XeHP.
- `config.{cpp,hpp}`, `host_scalars.hpp`, `utils.hpp`, `exec_types.hpp`, `conv.hpp`, `primitive.hpp` – Shared infrastructure.

`jit/`:
- `gen_kernel*.cpp/.hpp`, `pd.cpp/.hpp`, `walk_orders.hpp` – integration glue.
- `generator/` – nGEN code generator (umbrella + `pieces/`).
- `generator_dsl/` – higher-level DSL emitter.
- `dsl/` – DSL infrastructure.
- `include/gemmstone/` – public-ish headers (`problem.hpp`, `strategy.hpp`, `driver_info.hpp`, `kernel_catalog.hpp`, `kernel_selector.hpp`, `kernel_evaluator.hpp`, `microkernel/`, etc.).
- `selector/` – kernel selection / scoring + `db/*.db`.

`jit/generator/pieces/` – the actual emitter (see §2 table).

---

## 9. History notes

- The gemmstone generator originated as `kernel_generator` (XeHP era, hand-rolled fixedSystolic path) and was generalized into a strategy-driven framework supporting all Gen-architecture variants from Gen12LP through Xe3p.
- `raHW` indirection (`gen_kernel.cpp`) was introduced so PVC-tuned (XeHPC) strategies can be reused on Xe2/Xe3 without re-tuning the catalog.
- The int4-via-f16-bias trick (`planInt4ToF16`) replaces an earlier sequence using `and`+`shr`+`cvt`+`mul`; it cuts the int4→f16 dequant from 4 ops to 2 ops per element when followed by group-scale absorption.
- The Xe3p `upConvertATo8Bit`/`upConvertBTo8Bit` path was added when Xe3p shipped int8 DPAS but no int4 DPAS, making s4/u4 weights cheap-but-not-native.
- Stream-K and Hilbert/Boustrophedon walk orders were added to recover occupancy/L2 reuse on large dGPUs (XeHPC, BMG) where the simple linear walk leaves cache poorly utilized for skinny problems.
