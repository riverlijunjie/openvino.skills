---
name: dev_cm_gemm_optimization
description: Analyze, implement, benchmark, and optimize Intel C-for-Metal (CM) GEMM kernels, especially GGUF Q4_K/Q5_K/Q6_K XMX/dpas kernels, with architecture-aware tuning, roofline analysis, correctness validation, and interleaved A/B measurements.
---

# CM GEMM Kernel Optimization Skill

Use this skill when optimizing Intel CM (C-for-Metal) GEMM, dequantized GGUF GEMM, XMX/dpas kernels, cooperative SLM staging, register/GRF usage, work-group dispatch, or shape-dependent GPU performance.

This is a knowledge and methodology guide, not a repository map. It intentionally
contains no workspace-specific file paths. Locate the target CM source, host
harness, reference kernel, and benchmark scripts from the active workspace before
starting. Preserve CM semantics, compile with `-cmc`, and treat correctness,
register spills, occupancy, clock state, and dispatch/source synchronization as
first-class constraints.

## Fast bottleneck lookup

Start here when a new CM GEMM shape is slow:

| Symptom | Most likely cause | First experiment |
|---|---|---|
| `dpasonly` is much faster | dequant or operand path dominates | compare `nodecode`, `noAload`, and `noSLMrd` |
| `noAload` is much faster | activation load latency/traffic | inspect A layout, stride, 2D loads, and N-blocking |
| `noAtraf` is much faster | activation cache misses | increase A reuse, use row blocking, or repack A |
| `noSLMrd` is much faster | SLM consumer latency is exposed | test SLM layout/read width, not deeper buffering first |
| `nodecode` is much faster | dequant ALU is exposed | widen decode SIMD and amortize one decode over more dpas |
| performance collapses after a tile increase | register spill or occupancy loss | inspect spill diagnostics and live vector state |
| only long-K shapes are slow | A row stride/page locality | compare `TOKEN_GROUPS`, then consider K-blocked A repacking |
| only short-token shapes are slow | insufficient waves or padding | use a GEMV/slim path or a short-token-specific dispatch |
| sequential runs disagree wildly | clock/power/cache-state noise | use same-process interleaved A/B timing |
| output occasionally fails | SLM race or host/kernel mismatch | add fence-before-barrier and verify dispatch invariants |

The optimization order should normally be: **measure → classify the bottleneck →
remove redundant work → control registers/occupancy → tune reuse → inspect ISA →
consider a layout rewrite**. Do not start by changing dpas shape or increasing
GRF unless the measurements identify that as the limiting resource.

## 1. Non-negotiable workflow

1. Read the target `.cm`, its matching host harness, and the corresponding reference implementation before editing.
2. Identify the exact target architecture and device limits. Never assume BMG/Xe2 and PTL/Xe3 prefer the same tile.
3. Establish a correctness baseline before changing performance code.
4. Build all candidate variants from the same source and process whenever possible. Select variants with `-D` macros, not separate executable/process runs.
5. Benchmark candidates interleaved round-robin, with warmup and a cache-flush policy held constant. Use min-of-N for kernel-level A/B comparisons and report mean/median/min separately for sustained runs.
6. Inspect compiler output for `Spill memory used = ...`. A spill is normally a rejection unless a controlled measurement proves otherwise.
7. Record shape, quant type, token length, dispatch, macro configuration, register count, SLM usage, clock, latency, TFLOPS, roofline percentage, correctness status, and variance.
8. Re-test every adopted change on all relevant quant types and representative shapes. Do not generalize from one shape.
9. Update the kernel header comments and the repository memory with measured wins and rejected ideas so future work does not repeat dead ends.
10. Do not push commits to remote repositories unless explicitly requested.

## 2. Kernel and harness roles

For a CM GEMM optimization, identify four roles in the active workspace:

- the production CM kernel for each quantization format
- the host-side build and dispatch harness
- a correctness/reference implementation and input packer
- an interleaved A/B or component-ablation benchmark harness

Keep these roles conceptually separate. The kernel controls dataflow and
instruction generation; the host controls compile-time macros and dispatch; the
reference controls numerical validation; and the benchmark controls evidence.

The kernels use:

- fp16 activation input `[token_len, K]`
- SG-transposed GGUF weight layouts
- `cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, 8, TOKENS_PER_TILE>`
- `OPG=16`
- `TOKENS_PER_TILE=8`, fixed by the dpas repeat count
- fp32 accumulators and fp32 output
- cooperative weight dequantization through SLM in the current full path

The host must keep these values synchronized with the `.cm` macros:

- `ROW_GROUPS`
- `ROW_BLOCKS`
- `TOKEN_GROUPS`
- `TOKEN_LOCAL`
- `SLM_JBLK`
- `PF_A`
- `A_OUTER`
- any experimental `SLM_*`, `A_LOAD_2D`, `WALK_HBLK`, `DECODE_HALF`, or `CONS_DBUF` setting

For cooperative SLM, `TOKEN_LOCAL` is a correctness requirement, not merely a locality hint. The producer count and work-group size must agree:

\[
\text{SLM\_NPROD}=\text{SLM\_JBLK}\times\text{ROW\_GROUPS}\times\text{ROW\_BLOCKS}=\text{TOKEN\_LOCAL}
\]

The normal dispatch is:

\[
\begin{aligned}
\text{TOKEN\_SLOTS} &= \text{TOKEN\_LOCAL}/\text{ROW\_BLOCKS} \\
\text{TOKENS\_PER\_THREAD} &= 8\times\text{TOKEN\_GROUPS} \\
\text{ntiles} &= \lceil\text{token\_len}/\text{TOKENS\_PER\_THREAD}\rceil \\
\text{gsize} &= (N/(16\times\text{ROW\_GROUPS}\times\text{ROW\_BLOCKS}),\\
\quad \lceil\text{ntiles}/\text{TOKEN\_SLOTS}\rceil\times\text{TOKEN\_LOCAL}) \\
\text{lsize} &= (1,\text{TOKEN\_LOCAL})
\end{aligned}
\]

A wrong dimension-0 divisor produces missing/overlapping output and can look like a performance result. Always validate output before timing.

## 3. Hardware facts and architecture-specific defaults

### BMG / Arc B580 / Xe2

Measured characteristics:

- 160 XVE at approximately 2.9 GHz
- 8 threads per XVE
- 128 KB SLM per Xe-core
- approximately 456 GB/s GDDR6
- approximately 18 MB L2
- nominal fp16 peak around 116 TFLOPS
- hard thermal/power throttling under sustained runs

Adopted BMG configuration for the cooperative full kernels:

- `ROW_GROUPS=2`
- `TOKEN_GROUPS=4`
- `TOKEN_LOCAL=8`
- `ROW_BLOCKS=1`
- `SLM_JBLK=4` when required to keep the SLM producer count valid
- `PF_A=1` for the N-blocked path
- `A_OUTER=1`
- for long K (roughly `K >= 6144` at N around 4096), promote to `ROW_GROUPS=4/TOKEN_GROUPS=2` at the same `ROW_GROUPS*TOKEN_GROUPS` product (same accumulator size, same register count)

The BMG optimum follows the operand reuse model described in §7, but that model is only valid for the staging order it was derived under. `A_OUTER=1` is adopted here even though it costs registers relative to `A_OUTER=0` (120 to 128 in one measured case) -- see §6's note on why `A_OUTER` is not merely a register-pressure fix and why enabling it forces the §7 model to be re-derived.

### PTL / Arc B390 / Xe3 iGPU

Measured characteristics:

- 96 XVE at approximately 2.4 GHz
- 10 threads per EU/XVE
- 128 KB local memory/SLM per Xe-core
- approximately 16 MB L2
- shared LPDDR bandwidth around 136 GB/s
- nominal fp16 peak around 59 TFLOPS; dpas-only probe around 52–55 TFLOPS
- no useful GT-clock sysfs telemetry on the Windows remote box

Adopted baseline PTL configuration:

- `ROW_GROUPS=4`
- `TOKEN_GROUPS=2`
- `TOKEN_LOCAL=16`
- `ROW_BLOCKS=1`
- `SLM_JBLK=4`
- `PF_A=1`
- `A_OUTER=1`

`A_OUTER=1` holds activation tiles and streams B row groups, which is what makes `ROW_GROUPS=4` compile without spilling on Xe3's smaller per-thread register budget. An earlier PTL tile reached the same activation-traffic-per-dpas figure with `ROW_GROUPS=2/ROW_BLOCKS=2` (`256/(ROW_GROUPS*ROW_BLOCKS)` is identical either way); measurement showed `ROW_GROUPS=4/ROW_BLOCKS=1` consistently faster at the same product -- see §6 for why `ROW_GROUPS` should be preferred over `ROW_BLOCKS` whenever the register budget allows it. The smaller `TOKEN_LOCAL` also shrinks the SLM ring, so more work-groups stay resident per Xe-core.

### Never compare these defaults blindly

The same source is used for BMG and PTL, but the macro set is host-selected. The host harness should provide both automatic device selection and a force-config option for controlled testing.

## 4. Roofline and measurement discipline

### 4.1 Roofline calculation

Use separate compute and memory times and take the larger latency:

\[
t_{compute}=\frac{F}{P_{peak}},\qquad
 t_{memory}=\frac{B}{BW_{peak}},\qquad
 t_{roofline}=\max(t_{compute},t_{memory})
\]

The bound is compute-bound when `t_compute >= t_memory`, otherwise memory-bound. This is equivalent to the usual arithmetic-intensity ridge-point formulation, but is easier to audit.

For GEMM:

\[
F=2\times token\_len\times N\times K
\]

Use the quantized weight bytes plus activation and output bytes for the measured traffic estimate. State clearly whether percentages use nominal peak, dpas-only peak, mean latency, median latency, or min latency.

### 4.2 BMG clock drift

BMG throttles dramatically. A measured sequence of the same binary and shape was approximately:

- 1.72 ms at 2900 MHz
- 3.18 ms at 1467 MHz
- 4.74 ms at 1283 MHz
- 5.17 ms at 1150 MHz

Therefore two separate process runs are not valid before/after evidence. The first result can be several times faster than a later result solely because of clock state. The benchmark harness should sample GT clock after timed launches and report mean/median/min roofline values. If clock is below peak, report the clock and the normalized-at-peak estimate separately; do not silently mix them into the main comparison.

### 4.3 PTL sustained power limitation

PTL also shows sustained power effects: mean latency can grow with iteration count while min latency remains nearly stable. `--no-flush` can increase duty cycle and make sustained timing worse. Do not interpret a no-flush win/loss without interleaved order-controlled measurements.

### 4.4 Required A/B harness behavior

An interleaved A/B harness and a component-ablation harness are the reference pattern:

- build every variant from the same source in one process
- use the same input buffers and reference output
- run correctness once per variant
- warm up all variants
- alternate variants for every timed round
- flush consistently if the experiment calls for cold-cache behavior
- compare min-of-N for a kernel-level signal

If a sequential comparison changes an unchanged configuration, discard the comparison as clock/order noise.

Build a reusable component-ablation mode into the same harness (parametrized by which adopted configuration to probe from), not a one-off script: probes that remove the activation load entirely, remove only its cache traffic, remove the cooperative-decode SLM read, remove the dequant arithmetic, and remove all of the above (the dpas-only ceiling) are the standard set. Re-run this ablation after every structural change (a new staging order, a new N-blocking factor, a new architecture), not only once at the start of a kernel's life -- §6 gives a concrete case where the same ablation, re-run after one flag changed, showed a previously-dominant operand had become nearly free and invalidated the tuning model built around it. Treat one deliberately-wrong probe's numeric result as informative only for the component it isolates; a probe that patches out a component the kernel's register/format dependencies rely on can itself run slower than baseline, which is a broken probe, not a finding.

## 5. Compiler, CM, and ISA rules

### 5.1 Building

Standalone tests work with pyopencl and the Intel driver:

- `cl.Program(ctx, source).build(options='-cmc ...')`
- CM pointer arguments with `[[type("svmptr_t")]]` bind to ordinary `pyopencl` buffers

Useful compile options:

- `-cmc`
- `-DROW_GROUPS=N`
- `-DTOKEN_GROUPS=N`
- `-DTOKEN_LOCAL=N`
- `-DROW_BLOCKS=N`
- `-DA_OUTER=N`
- `-DSLM_JBLK=N`
- `-DPF_A=N`
- `-Qxcm_register_file_size=N` for controlled GRF experiments

After changing an OpenVINO `.cl` kernel, force codegen regeneration (`touch`, run the OCL codegen target, and refresh the generated kernel database) before trusting a build. For CM standalone files, ensure the source used by `cl_src()` is the edited file.

### 5.2 Register and spill screening

`-mCM_printregusage` is useful on BMG and `ocloc`, but is silently ignored by the Windows PTL driver. On PTL, use timing plus compiler spill warnings. A warning such as `Spill memory used = 7040 bytes` is a strong rejection signal.

Do not use `-Qxcm_register_file_size=256` merely because it allows a larger tile. It halves resident threads/XVE (8 to 4 on BMG), and this kernel is latency-sensitive. Experiments with GRF=256 and larger tiles were a wash or a loss.

### 5.3 CM syntax/API pitfalls

- CM has no OpenCL sub-group lanes. One CM thread computes a complete vector; do not port an OpenCL subgroup dimension literally.
- Use `uint` rather than `size_t` for pointer offsets.
- `cm_ptr_load<T,N>` and `cm_ptr_store<T,N>` require 4- or 8-byte element types. For half/ushort/char, load as `uint32` and reinterpret with `.format<half>()`, `.format<ushort>()`, or `.format<char>()`.
- Callee output parameters need `vector_ref<T,N>` or `matrix_ref<T,R,C>`, not ordinary C++ references to CM vectors.
- `cm_exp()` is base-2 exponential. Multiply by `log2(e)` for natural exponential.
- `cm_sum()` is a horizontal reduction.
- Arrays of CM vectors compile and are useful for explicitly bounded register state.
- A single vector object is limited to less than 8192 bytes. Larger token tiles need arrays of smaller vectors, but arrays can still spill.
- Pair `cm_slm_fence(CM_LOCAL_BARRIER)` with `cm_barrier()` after SLM writes before SLM reads. A bare barrier produced a real nondeterministic reduction race.
- Flat vector stores/loads (`cm_ptr_store<T,N>`/`cm_ptr_load<T,N>`) are limited to a maximum element count (commonly 64) and to legal LSC vector-size values; a merged store spanning several N-blocking groups can silently stop compiling once the group count grows (both from exceeding the element-count limit and from landing on a non-power-of-two size). Guard a merged-store optimization with a compile-time size check and fall back to per-group stores above the limit instead of assuming any N-blocking factor is safe.

### 5.4 ISA inspection

Use `ocloc` and IGC shader dumps when available:

- register/assembly count options such as `-Qxcm_print_asm_count`
- `IGC_ShaderDumpEnable=1`
- `IGC_DumpToCustomDir=<directory>`

Read the hot loop, not just total instruction count. On PTL, the hot loop had only about 13–15% dpas instructions and roughly 360 non-dpas instructions, including dequant bit-field/type-convert/VNNI scatter operations. The dpas issue stream was already close to optimal; the remaining gap was operand load latency and architecture limits rather than a missing dpas instruction.

## 6. Optimization history: adopted changes

The following sequence is the validated history for the full Q4_K/Q5_K/Q6_K CM kernels. Preserve these conclusions unless new hardware or a materially different data layout invalidates them.

### Foundation: port and baseline validation

The first CM work established a standalone pyopencl workflow and ported the
validated GGUF dequantization/layout logic from the SYCL/OpenCL reference. The
SG-transposed weight layout was retained because each fixed `(h, block, j,
chunk)` contains 16 contiguous lane values, which maps naturally to CM vector
loads. CM has no OpenCL subgroup-lane dimension: an OpenCL OPG=16 subgroup
becomes one CM thread computing a full vector.

The early full GEMM version was intentionally conservative: it used the real
`cm_dpas` engine but retained K-split and SLM accumulation/reduction. It was
correct but only around 5–10% of the BMG dpas roofline. The initial bottleneck
analysis established that the problem was not numerical accuracy or dpas
throughput; dequant work, redundant activation/weight movement, barriers, and
register pressure dominated. This baseline is useful as a correctness oracle,
but it must not be reintroduced as a performance design.

The port also uncovered two reusable correctness rules. First, every SLM
producer/consumer sequence needs a local fence before its barrier; a bare
`cm_barrier()` caused a nondeterministic Q6_K reduction race. Second, CM
compiler output must be treated as part of the test result: a compile that
silently spills may remain numerically correct while becoming several times
slower.

### v4 baseline to v5: amortize dequant over token groups

The original full path decoded a K=32, N=16 VNNI tile pair and consumed it for only one 8-token dpas pair. Dequant instructions dominated the kernel.

Adopted changes:

1. Remove K-split and SLM accumulator reduction. One thread owns all K and keeps accumulators in registers.
2. Stream activation slices instead of keeping a 4 KB `[8 tokens, 256 K]` A tile in registers.
3. Reuse a decoded B tile for `TOKEN_GROUPS` dpas pairs.
4. Use register-free cache prefetch rather than loading future operands into registers.

The v5 path produced roughly 2.0–3.2x improvements in the measured BMG cases. The conceptual gain is dequant amortization; the structural enabler is removing K-split and reducing live activation state.

### v6: LSC 2D activation block load

Replace multiple flat activation loads, register moves, and manual bounds clamps with:

`lsc::block_2d_desc<half, 2, 8, 16>`

The two X blocks return `[Alo | Ahi]` in dpas operand order. Hardware surface-height clamping handles tail tokens. This measured approximately 1.83–1.96x on top of the prior path and removed enough register pressure for Q5_K to use `TOKEN_GROUPS=4` without spilling.

Constraints:

- `TOKENS_PER_THREAD = 8 * TOKEN_GROUPS`
- one 2D descriptor is limited to 32 rows, so larger activation prefetches must be split
- do not replace the descriptor with two manually targeted flat loads without measuring; that variant was slower

### v7: cooperative SLM weight decode

Threads in a work-group share the same weight row group. Previously every thread decoded the same eight sub-blocks. Cooperative staging assigns decode work to producers, writes decoded VNNI B tile pairs to an SLM ring, and lets all consumers read them.

Adopted BMG shape:

- `TOKEN_GROUPS=4`
- `TOKEN_LOCAL=8`
- `SLM_JBLK=8`
- `SLM_DEPTH=1`
- `SLM_SLOTS=2`
- full barrier protocol
- 16 KB SLM/work-group
- no extra cache prefetch in the cooperative path

Measured BMG result: approximately 60–73 TFLOPS, or 52–63% of nominal 116 TFLOPS dpas roofline, versus approximately 10% for the earlier version. Cumulative v4-to-v7 improvements were approximately 5.6x q4k, 6.5x q5k, and 7.5x q6k in the measured cases.

Critical barrier lesson: a split SDPA-style barrier needs at least `SLM_DEPTH+2` slots. `SLM_DEPTH=1` with the split barrier was silently wrong. The plain full barrier needs `SLM_DEPTH+1` slots and was faster because it preserved occupancy.

### v8: N-blocking with ROW_GROUPS

The activation tile does not depend on the output row group. Give one thread multiple adjacent row groups and put the row-group loop inside the token-group loop so one A register feeds multiple dpas pairs.

Adopted BMG configuration:

- `ROW_GROUPS=2`
- `TOKEN_GROUPS=4`
- `TOKEN_LOCAL=8`
- `SLM_JBLK=4` to preserve the 16 KB SLM ring
- `PF_A=1` after remeasurement

Measured speedups versus `ROW_GROUPS=1` included approximately 1.39x q4k, 1.25x q5k, and 1.14x q6k on large 4096x4096 shapes with long token lengths.

The host dimension-0 dispatch must divide by `OPG*ROW_GROUPS`; otherwise output coverage is wrong.

### A_OUTER structural rewrite for PTL

`A_OUTER=0` holds B tiles for all row groups and streams A. At larger `ROW_GROUPS`, live B grows and spills. `A_OUTER=1` holds the token-group activation tiles and streams one row group of B at a time. Live B becomes approximately 1 KB independent of `ROW_GROUPS`.

This rewrite reduced PTL register pressure from roughly 119 to 102 registers for the adopted configuration and improved PTL timings by approximately 2–5% in the tested cases. It is particularly important when screening larger `ROW_GROUPS` configurations.

### PTL ROW_BLOCKS and wider cooperative groups

PTL has 10 threads/EU and shared LPDDR, so its best configuration is not BMG's. `ROW_BLOCKS=2` divides a work-group into row-blocks and token slots:

- `rb = lid % ROW_BLOCKS`
- `ts = lid / ROW_BLOCKS`
- threads with the same token slot use the same A addresses
- dimension 0 becomes `N/(OPG*ROW_GROUPS*ROW_BLOCKS)`

The adopted PTL result is:

- `ROW_GROUPS=2`
- `TOKEN_GROUPS=2`
- `TOKEN_LOCAL=32`
- `ROW_BLOCKS=2`
- `SLM_JBLK=8`
- `PF_A=0`
- `A_OUTER=1`

The 13-shape PTL sweep with `--warmup 100 --iters 1000` improved from approximately 39.7% mean / 41.0% median / 44.6% min to 42.2% / 43.5% / 48.0% across the reported roofline.

### Shape-aware PTL tuning for long K

Long-K shapes are asymmetric because A is row-major in K. The eight rows in one activation 2D load are separated by `K*2` bytes. The load spans approximately 65 KB at K=4096 and 196 KB at K=12288, increasing TLB and page/row locality pressure on PTL's shared memory system.

The important observation is directional:

- `K=12288, N=4096` is slow
- `K=4096, N=12288` is materially healthier even though the weight matrix has the same total byte size
- therefore this is not simply L2 capacity

Interleaved PTL measurements of `TOKEN_GROUPS=4` versus the normal `TOKEN_GROUPS=2` found:

- K=16384: 1.08–1.14x
- K=12288: 1.11–1.21x across q4k/q5k/q6k
- K=10240: approximately 1.20x
- K=8192: approximately 0.95x
- K=6144: approximately 0.95x
- K=4096: approximately 0.79–0.96x
- K=2048: approximately 0.90x

The final host rule is `LONG_K=10240`: promote PTL from `TOKEN_GROUPS=2` to `TOKEN_GROUPS=4` only when `K >= 10240`. BMG already uses `TOKEN_GROUPS=4`, so the rule is a no-op there. `--no-shape-tune` disables it for controlled A/B measurements.

This is an in-kernel mitigation, not a complete fix. The structural fix is to repack A in a K-blocked layout in the caller so token rows within a K block are contiguous and the address stride no longer depends on full K.

**Shape-tuning rules can conflict once the underlying tile changes.** A rule derived under one adopted tile ("promote `TOKEN_GROUPS` for long K") can become harmful once a different rule is layered on top ("promote `ROW_GROUPS`" on the same shapes), because the two together can exceed the register budget. Guard shape-tuning rules against each other explicitly (e.g. only fire the older rule when the newer one has not already changed the tile), and re-measure whether an older rule is still the best available option rather than assuming it still applies.

### Small-block quantization formats reuse the K-type sub-block decoder

GGUF quant formats built on a single 32-weight block with one scale (and
optionally one offset) decode with exactly the arithmetic shape of ONE
sub-block of a K-type format's cooperative decode (K-type formats group eight
32-weight sub-blocks into one 256-wide K-block). Concretely, if a K-type
sub-block decodes as `w = q*scale - minv`, then:

- a format with one scale and an implicit fixed zero point maps to
  `scale = d, minv = zero_point*d`
- a format with an explicit scale and offset maps to `scale = d, minv = -m`
- a format with a signed wide (e.g. int8) quant and one scale maps to
  `scale = d`, with no bit-field extraction, only a wider strided byte read

This means the entire cooperative-decode dataflow (SLM ring, VNNI tile
scatter, `ROW_GROUPS`/`TOKEN_GROUPS` N-blocking, `A_OUTER` staging) can be
reused verbatim for the new format; only the producer's metadata load and
the scale/offset arithmetic feeding the shared bit-field-extraction-and-scatter
routine change. Do not re-derive the dataflow or re-run the full ablation
history for such a format from scratch -- port the decode front end onto the
existing structure first, then re-measure the SAME configuration sweep,
since the new format's byte size (payload and metadata) can shift the
operand-traffic balance even when the dataflow is unchanged (a wider raw
payload, for example, increases the weight side of the traffic model without
changing the activation side at all, so a config screen can still surface a
different optimum for it).

### A_OUTER is a latency-hiding technique on compute-rich parts too, not just a register-pressure fix

`A_OUTER` was originally adopted to solve register pressure on a
register-poor architecture: holding the activation tiles and streaming one
weight tile at a time keeps live weight state constant instead of scaling
with `ROW_GROUPS`. On a compute-rich, register-adequate architecture the same
flag can still win, but for a DIFFERENT reason and despite COSTING registers:
issuing every independent activation load back-to-back before the first dpas
(instead of interleaving one load into the middle of every dpas chain) lets
their latency overlap compute. A component-ablation probe that removes the
SLM read entirely is the tell: if it comes back nearly free once `A_OUTER=1`
is on, the flag has already hidden that cost behind the batched activation
loads, and its old justification (register pressure) is no longer the reason
it wins. Do not assume a flag's mechanism transfers between architectures
just because the flag transfers -- re-run the ablation after enabling it on a
new part.

### Re-deriving the operand-traffic model after a structural change

An operand-traffic cost model derived under one dataflow (for example
`bytes/dpas = 256/ROW_GROUPS + 512/TOKEN_GROUPS`, which weights the
cooperative-SLM operand roughly twice as heavily as the activation operand)
is only valid as long as both operands are still equally exposed. Once a
structural change hides one operand's latency (see the `A_OUTER` note above),
re-run the component ablation before trusting the old model: if the hidden
operand's removal is now nearly free, its term effectively drops out of the
model, and the optimum shifts toward maximizing reuse of the operand that is
still exposed -- usually meaning the largest `ROW_GROUPS` (or equivalent
N-blocking factor) that the accumulator/register budget allows, even though
the old model would have predicted a smaller value. Confirm the new optimum
by sweeping configurations that hold the accumulator size constant
(`ROW_GROUPS*TOKEN_GROUPS` fixed) while trading `ROW_GROUPS` for
`TOKEN_GROUPS`, and re-measure across every quant format and representative
shape, not just the one that motivated the change -- the pay-off is usually
shape-dependent (largest on the shapes where the now-cheaper operand used to
dominate) and can be a small loss elsewhere, which is exactly what a
shape-tuning threshold should gate on.

### ROW_GROUPS beats ROW_BLOCKS at an equal traffic budget

`ROW_BLOCKS` and `ROW_GROUPS` can both cut the activation-traffic-per-dpas
figure by the same factor (`256/(ROW_GROUPS*ROW_BLOCKS)` does not care which
factor supplies the reduction), so it is tempting to treat them as
interchangeable. They are not: `ROW_BLOCKS` gets its reuse by having several
threads in a work-group issue the SAME activation address and hoping the
second and later ones hit in L1, while `ROW_GROUPS` gets its reuse for free
by holding one activation value in registers and feeding it to multiple dpas
instructions directly. At an equal product, prefer the larger `ROW_GROUPS`
(enabled by `A_OUTER` if needed) over `ROW_BLOCKS` -- it was consistently
faster in measurement, and it also lets `TOKEN_LOCAL` (and therefore the
cooperative-decode SLM ring) shrink, which increases resident work-groups per
core. Reserve `ROW_BLOCKS` for cases where the register budget genuinely
cannot support a larger `ROW_GROUPS`.

### Weight layout consistency across a kernel family

When a family of related quant-format kernels is maintained alongside a
production integration that repacks weights at compile time, keep exactly ONE
weight layout per format across the whole family, matching whatever the
production repack emits -- even if a standalone variant (for example,
interleaving two small per-row metadata fields into one wider load to shave
a producer message) measures a small, inconsistent win in isolation. A single
maintained layout is worth more than a small, format-specific gain: it
removes an entire class of layout-drift bugs between the standalone kernel,
its test harness's weight packer, and the production repack pass, and it
means a correctness fix or layout audit only has to be done once. Confirm the
two layouts are identical by reading the production repack's byte offsets and
lane-interleave order directly (do not just compare kernel source), since the
production repack is the actual source of truth the kernel will be fed from
in deployment.

## 7. Analytical model and dataflow rules

With the row-group loop inside the token-group loop, measured operand traffic per dpas is approximately:

\[
bytes/dpas = 256/ROW\_GROUPS + 512/TOKEN\_GROUPS
\]

The accumulator footprint is approximately:

\[
acc\_bytes = ROW\_GROUPS\times TOKEN\_GROUPS\times 512
\]

On BMG, the useful product is constrained to roughly `ROW_GROUPS*TOKEN_GROUPS <= 8`, and the traffic model predicts:

- `(1,8)`: 320 B/dpas
- `(2,4)`: 256 B/dpas — best
- `(4,2)`: 320 B/dpas
- `(8,1)`: 544 B/dpas

Measured variants followed this model. The optimal dataflow is not the one with the smallest accumulator alone; operand reuse dominates.

This model assumes the cooperative-SLM operand and the activation operand are both equally exposed to latency. That assumption breaks once a staging change (see §6's `A_OUTER` notes) hides one operand behind the other. When a component ablation shows one operand's removal is nearly free, drop its term and re-fit: for example, once the SLM term is effectively free the model collapses to

\[
bytes/dpas \approx 256/(ROW\_GROUPS \times ROW\_BLOCKS)
\]

and the optimum moves to the largest `ROW_GROUPS*ROW_BLOCKS` (preferring `ROW_GROUPS`, per §6) that keeps `ROW_GROUPS*TOKEN_GROUPS` inside the accumulator budget. Always re-validate a model after changing which operand a structural change exposes; do not keep optimizing against a stale model.

K-splitting was rejected analytically and experimentally:

- it leaves operand traffic unchanged
- it adds synchronization and reduction
- the grid has adequate parallelism at normal token lengths
- variants that traded reuse for more thread tiles lost
- it may only make sense for tiny token lengths, where GEMV/slim kernels are already the intended path

Do not spend freed accumulator registers on an explicit consumer SLM double buffer without a new measurement. `CONS_DBUF` was neutral-to-worse because operand reuse dominated and the compiler already overlapped the unrolled SLM reads.

## 8. Experiments that were rejected

Do not repeat these without a new architecture, data layout, compiler, or workload condition.

### Register and tile experiments

- `-Qxcm_register_file_size=256`: 0.87–1.08x in combined tile experiments; reduced resident threads and did not beat the default.
- BMG `TOKEN_GROUPS=5/6/8`: spills or under-amortizes work.
- PTL `TOKEN_GROUPS=8`: approximately 0.15–0.18x with a 7040-byte spill.
- PTL `ROW_GROUPS=4` without A_OUTER: spills; with A_OUTER it compiled but was still approximately 0.86–0.95x.
- PTL `ROW_BLOCKS=4/8`: approximately 0.83–0.90x / 0.59–0.76x.
- PTL `TOKEN_LOCAL=64/128`: slower; 128 can fail with `INVALID_WORK_GROUP_SIZE`.
- GRF=96/64/128 on PTL: 96 slower, 64 spills badly, 128 approximately neutral.

General rule: for a latency-bound kernel, resident threads hide latency; extra registers do not automatically help.

### Prefetch and walker experiments

- Loading future A/B tiles into registers for software pipelining: 2.3x slower and spills 2048–3648 bytes.
- `PF_W` and deeper `PF_A`: generally neutral or worse in the cooperative path; BMG's `PF_A=1` only became useful after N-blocking.
- `WALK_HBLK=2/4/8/16`: 0.26–1.08x on skewed PTL shapes. The swizzle breaks the hardware prefetcher's sequential stream.
- SLM depth 2 / four-slot ring: approximately 0.72–0.80x because 32 KB SLM halves occupancy.

Use cache prefetch instructions that do not allocate destination GRF; do not turn prefetch into live data.

### Operand format and decode experiments

- Predecode weights to fp16 VNNI: 0.69–0.93x. fp16 weights are about 3.5x larger than Q4_K, so memory traffic costs more than decode+SLM removal saves.
- `DECODE_HALF=1`: roughly neutral, q4k 0.96–0.99x, q5k 0.99–1.02x, q6k 1.00–1.07x; it adds rounding differences and does not remove the accumulator/operand register wall.
- One 1 KB SLM read instead of two 512 B reads: 0.99–1.00x; SLM latency, not message count, is the issue.
- Activation cache hints: default/cached were neutral; uncached was 0.91–0.95x, confirming that L1 helps.
- Splitting a 64 B A load into two 32 B flat loads: approximately 0.9x; the LSC 2D load is better.
- SLM padding: generally 0.74–0.79x on PTL.

### K-split and smaller accumulator

Measured smaller-accumulator plus `CONS_DBUF` variants lost approximately 0.56–0.96x depending on shape. The algebraic reuse model and dpas-only probes show that K-split cannot recover enough to justify its reduction overhead for the full GEMM path.

### Consolidated loads and descriptors (after enabling A_OUTER)

- One wide 2D block load consolidating several per-token-group activation loads into a single descriptor read: fewer LSC messages and fewer live registers, but roughly 0.95–1.00x. This confirms the remaining cost is L2 traffic and latency, not message count or register pressure, once `A_OUTER` is already hiding the load latency.
- One 2D descriptor per token group (to avoid repeatedly re-pointing a shared descriptor at each group's row range): 0.67–0.71x. The extra descriptor state spilled; a cheap-looking way to remove "redundant" scalar descriptor updates is not free if it grows live register state.
- A deeper activation prefetch distance after `A_OUTER` is already staging loads ahead: approximately 0.93–0.97x, i.e. a small loss -- the batched loads already provide the latency hiding a deeper prefetch would add.
- Stacking two independent shape-tuning promotions (a larger N-blocking factor AND a larger token-group factor) on the same shape without checking their combined accumulator/register footprint: spills and regresses sharply (as low as approximately 0.2x in one measured case). Always check that compounded shape rules stay inside the same register budget as each rule does alone.

### Unsupported ideas

- `cm_dpasw` is not available for BMG/Xe2; compile probes report `CM_HAS_DPASW is NOT defined for this target`.
- `A_LOAD_2D=0` does not compile in the `COOP_SLM` branch because the flat A path is only implemented in the non-cooperative branch.

## 9. Bottleneck diagnosis checklist

When a new shape is slow:

1. Confirm tuple interpretation: `(quant, K, N, token_len)`, not `(N,K,token_len)`.
2. Check `K % 256 == 0` and `N % (OPG*ROW_GROUPS*ROW_BLOCKS) == 0`.
3. Print the selected macros and dispatch. Verify `TOKEN_LOCAL`, `ROW_BLOCKS`, and `SLM_JBLK` are consistent.
4. Compare K and N independently. If K is large and N is moderate, inspect A stride and page locality before blaming weights/L2.
5. Compare `TOKEN_GROUPS=2` and `4` interleaved. For PTL, test the long-K threshold around K=8192/10240/12288.
6. Check for compiler spill warnings.
7. Run the same shape through the dpas-only, no-A-load, no-A-traffic, no-SLM-read, and no-decode ablations to locate the dominant path.
8. Inspect clock state on BMG and iteration-count/power effects on PTL.
9. Only then consider new tiling or dataflow changes.

Typical interpretation of ablations:

- `dpasonly` much faster: non-dpas operand/dequant path dominates.
- `noAload` much faster: A load latency/traffic dominates.
- `noAtraf` much faster: A cache traffic dominates; consider A staging/repacking.
- `noSLMrd` much faster: SLM consumer latency is exposed after N-blocking.
- `nodecode` much faster: dequant arithmetic is exposed, especially on PTL.
- all variants close: likely dispatch, clock, cache state, or measurement noise.

## 10. Correctness requirements

For every variant:

- use the same quantized input and activation seed as the baseline
- compare against a NumPy reference built from the exact quantized layout
- round both dequantized weights and activations to fp16 before the fp64 reference matmul when matching the fp16 dpas path
- test tail token lengths, including non-multiples of `TOKENS_PER_THREAD`
- test K/N in both square and skewed forms
- require `ALL PASS` before interpreting performance
- after any SLM protocol change, repeat the same case many times to detect nondeterministic races

The `cm_slm_block_write -> fence -> barrier -> cm_slm_block_read` ordering is mandatory for producer/consumer reductions and cooperative staging. A result that passes once is not sufficient.

## 11. Recommended experiment templates

### Fast local correctness check

Use the normal correctness/benchmark harness with a small iteration count and explicit shapes. Verify the selected tile parameters, dispatch dimensions, and `ALL PASS`.

### Interleaved A/B

Add named variants to the A/B harness with explicit `(RG,TG,TL,extra_options)` entries. Keep the first variant as baseline. Always update dispatch calculations and row-block parsing for `ROW_BLOCKS` experiments.

### PTL remote testing

Use the existing parameterized remote helper and keep credentials private. Sync only the required files. Run correctness before long sweeps. Use the user's requested `--warmup 100 --iters 1000` for final sustained numbers, but use shorter interleaved sweeps to screen candidates.

### ISA/compiler screen

For each candidate record:

- register count when the compiler reports it
- spill bytes
- SLM bytes per work-group
- work-group size and estimated resident work-groups
- dpas count and load/send count if assembly is available
- whether the hot loop is latency- or throughput-limited

## 12. Future structural opportunities

Configuration tuning cannot fully solve the PTL long-K problem. The next substantial optimization should be caller-side A repacking:

- pack activations by K block, for example `[K_block][token][within_block_K]`
- make the eight-token 2D load contiguous or nearly contiguous inside a 256-K block
- adapt the kernel descriptor and dispatch to the packed layout
- compare repacked and original paths using the same interleaved methodology
- validate all token tails and integration call sites

A larger rewrite could stage A in SLM for a 2D work-group tile, but it must account for 128 KB SLM, occupancy, synchronization, and the existing cooperative B decode ring. Do not increase SLM depth without calculating resident work-groups first.

Other ideas worth considering only with new evidence:

- architecture-adaptive `REG_N` rather than hardcoded OPG where the target supports it
- hardware VNNI load for already-decoded weights, if the storage layout permits it
- caller/kernel co-design for activation packing
- a separate short-token GEMM/GEMV dispatch rather than forcing full GEMM at tiny token lengths

## 13. Completion standard

A CM GEMM optimization is complete only when:

- the source and host dispatch are synchronized
- all targeted quant types pass correctness
- tail and skewed shapes pass correctness
- no unintended compiler spill exists
- performance is measured interleaved against a baseline
- clock/power/cache state is reported
- roofline bound and peak assumptions are explicit
- rejected experiments and their data are documented
- the repository memory is updated
- no unrelated files or remote commits were changed
