# DPAS GEMM Kernel Optimization Guide (Intel Xe2 / BMG / B580)

## Final Result: ~62 TFLOPS / **~65%** efficiency on K=2560, **~60%** on K=4096 (96 TFLOPS peak)

This document summarizes 28 iterations of optimizing an OpenCL f16 GEMM kernel on Intel Arc B580 (BMG, Xe2 architecture).

---

## 1. Performance Progression

| Version | Strategy | TFLOPS | Efficiency | Delta |
|---------|----------|--------|------------|-------|
| v1 | VNNI SLM + scalar cooperative load | 3.4 | 3.5% | baseline |
| v2 | Row-major SLM + block_read global | 14.4 | 15% | +11.5% |
| v6 | 16 subgroups, WG_TILE=128×64, SLM | 23.0 | 24% | +9% |
| v10 | No SLM, direct global→DPAS | 16.0 | 16.7% | regression |
| **v11** | **2D block read (8r16x1c + transform)** | **33.6** | **35%** | **+11%** |
| v12 | Double-width B (16r16x2c transform) | 44.2 | 46% | +11% |
| v13 | Full 32r16x1c A read | 45.1 | 47% | +1% |
| v16 | WG_M=2, WG_N=4 (8 sg) | 45.8 | 47.7% | +0.7% |
| v18 | WG_M=4, WG_N=4 (16 sg, 256 thr) | 48.4 | 50.4% | +2.7% |
| **v19** | **WG_M=4, WG_N=8 (32 sg, 512 thr)** | **52.8** | **55%** | **+4.6%** |
| v20 | WG_M=8, WG_N=8 (64 sg, 1024 thr) | 50.2 | 52.3% | -2.7% |
| v22 | K-unroll×2 (separate A/B buffers per k-step) | 54-55 | 56-57% | +1-2% |
| v23 | K-unroll×2 + WG_M=8, WG_N=4 | 50-51 | 52-53% | regression |
| **v24** | **K-unroll×2 + 2D wide A read `32r16x2c`** | **55** | **57.4%** | **+2.4%** |
| **v27/v28** | **v24 + WG swizzle (SWIZZLE=2) for L3 locality** | **~62 / 57.5** | **~65% / ~60%** (K=2560 / K=4096) | **+7-10%** |

### Failed Experiments

| Version | Strategy | TFLOPS | Why It Failed |
|---------|----------|--------|---------------|
| v14 | K-unroll×2 + prefetch | 37.2 | Register pressure from double buffers |
| v15 | Software-pipelined double buffering | 42.9 | Same: register pressure > latency hiding |
| v17 | TILE_M=64, 16 accumulators | 19.2 | Register spills (128 GRFs insufficient) |
| v20 | 64 subgroups / 1024 threads | 50.2 | Occupancy drops (only 2 WGs/XeCore) |
| v21 | `intel_sub_group_2d_block_prefetch_*` | n/a | CL_INVALID_WORK_GROUP_SIZE (-54): prefetch builtins unsupported / incompatible |
| v25 | `intel_sub_group_2d_block_read_transform_16b_32r16x2c` (wide B transform) | n/a | Compiles, but data layout differs from `16r16x2c`; max_rel_err > 3000 (correctness fail) |
| v26 | K-unroll×4 (BLOCK_K=64, 2 A + 4 B in flight) | 28 | Register spill: 6 buffers per iter exceed safe GRF budget |
| v27 SWIZZLE=8 | Larger swizzle group | 54 (K=4096) | Group too wide → straddles L3 capacity, hurts K=4096 |
| v29 | SLM cooperative B load (sg_row=0 fetches, broadcasts via SLM) | 40-43 | 2 WG-barriers per iter (`barrier(CLK_LOCAL_MEM_FENCE)` × 2 at 512 thr/WG) cost ~23 percentage points — wipes out L3-savings gain. Sub-WG sync not available in OpenCL. |
| probe | `intel_sub_group_2d_block_read_16b_32r16x4c` (wider A) | n/a | Builtin not defined on B580; only `*1c` and `*2c` available for `32r16` shape |

---

## 2. Critical Optimization Insights

### 2.1 Use 2D Block IO Instead of SLM (24% → 35%, biggest single win)

**Problem**: SLM-based GEMM requires cooperative loads, barriers, and bank-conflict-free access patterns. This adds overhead and limits occupancy.

**Solution**: `cl_intel_subgroup_2d_block_io` provides hardware-accelerated strided 2D reads directly from global memory into register-ready formats.

```c
// A: reads 32 rows × 16 cols (one BLOCK_K) in a single message
ushort a_data[32];
intel_sub_group_2d_block_read_16b_32r16x1c(
    (__global void*)A, width_bytes, height, pitch_bytes,
    (int2)(col_byte_offset, row), a_data);

// B: reads 16 rows × 32 cols with VNNI transform in one message
uint b_data[16];
intel_sub_group_2d_block_read_transform_16b_16r16x2c(
    (__global void*)B, width_bytes, height, pitch_bytes,
    (int2)(col_byte_offset, row), b_data);
```

**Key constraints**:
- Base pointer must be 64-byte aligned
- `width_bytes` ≥ 64, must be actual total row width in bytes
- `pitch_bytes` ≥ width_bytes and multiple of 16
- `coord.x` must be multiple of 2 for 16-bit data (byte offset / sizeof(half))
- Subgroup size must be 16

### 2.2 Maximize DPAS Per Load Message (35% → 46%)

**Principle**: The ratio of DPAS instructions to load messages is the key metric. More DPAS per message = higher XMX utilization.

| Config | Messages/k_step | DPAS/k_step | Ratio |
|--------|----------------|-------------|-------|
| v11 (4×A + 1×B) | 5 | 8 | 1.6 |
| v12 (4×A + 1×B_wide) | 5 | 8 | 1.6 |
| v13 (1×A_full + 1×B_wide) | **2** | **8** | **4.0** |

The `32r16x1c` read for A and `16r16x2c` transform read for B achieve the optimal 2 messages → 8 DPAS ratio.

### 2.3 Workgroup Size Sweet Spot: 512 Threads (47% → 55%)

**Why larger WGs help**: Subgroups within the same WG access overlapping A/B tiles, creating natural L3 cache reuse without explicit coordination.

**Why too large hurts**: B580 has 128 HW threads per XeCore. With WG_SIZE=1024 (64 subgroups), only 2 WGs fit per XeCore → poor latency hiding when all threads stall on memory.

| WG Config | Threads | WGs/XeCore | Efficiency |
|-----------|---------|------------|------------|
| WG_M=2, WG_N=4 | 128 | 16 | 47.7% |
| WG_M=4, WG_N=4 | 256 | 8 | 50.4% |
| **WG_M=4, WG_N=8** | **512** | **4** | **55%** |
| WG_M=8, WG_N=8 | 1024 | 2 | 52.3% |

**Rule of thumb**: Target 4 WGs per XeCore (512 threads × 4 = 2048 ≈ 128 HW threads × 16 ... actually 512/16=32 sg, 32 threads per WG effective for scheduling). The key is balancing L3 reuse (bigger WG) vs occupancy (more WGs).

### 2.4 Register Budget Is the Hard Limit

On Xe2, each thread has **128 GRF × 32 bytes = 4096 bytes** of register space.

Our v19 kernel uses:
- 8 × float8 accumulators = 8 × 32B = 256B
- A data (ushort[32]) = 64B
- B data (uint[16]) = 64B
- Misc (pointers, loop vars) ≈ 64B
- **Total ≈ 448B** — well within budget

v17 tried 16 float8 accumulators (512B) + larger A/B buffers → **register spills → 20% efficiency**.

**Lesson**: Never exceed ~8 float8 accumulators. The compiler will spill to memory silently.

### 2.5 Software Pipelining Doesn't Help Here

Unlike CPU, the GPU hardware scheduler already hides memory latency by switching between threads. Adding explicit double-buffering:
- Doubles register usage for load buffers
- May cause register spills
- Provides no additional latency hiding (HW already does this)

This was confirmed by v14 (38.7%) and v15 (44.7%) both being worse than v13 (47%).

### 2.6 Prefetch Instructions Are Counterproductive

`intel_sub_group_2d_block_prefetch` was tested in v14. Result: worse performance. The load/store unit has limited bandwidth — prefetch messages compete with actual load messages for the same resources.

**When prefetch might help**: Only if the k-loop body is long enough (many DPAS between loads) that next-iteration data wouldn't naturally be in flight. With only 2 loads + 8 DPAS per iteration, the pipeline is already saturated.

---

## 3. B580 Hardware Parameters (for Roofline Analysis)

| Parameter | Value |
|-----------|-------|
| GPU Frequency | 2400 MHz |
| XeCores | 20 |
| EUs per XeCore | 16 |
| HW Threads per EU | 8 |
| HW Threads per XeCore | 128 |
| Total HW Threads | 2560 |
| XMX f16 Peak | 96 TFLOPS |
| FPU-only f16 Peak | ~12 TFLOPS |
| Memory Bandwidth | 456 GB/s |
| L3 Cache | 32 MB |
| SLM per XeCore | 128 KB |
| GRF per Thread | 128 × 32B = 4096B |
| Subgroup Size | 16 |

**Arithmetic Intensity Threshold**: 96 TFLOPS / 456 GB/s = **210 FLOP/byte**. For GEMM with M=N=K=2048: AI = 2×2048³ / (3×2048²×2) = 2048/3 ≈ 683 FLOP/byte → **compute-bound** (good).

---

## 4. DPAS Instruction Reference

```c
// f16 × f16 → f32, systolic depth 8, repeat count 8
// Computes C[8,16] += A[8,16] × B[16,16]
float8 intel_sub_group_f16_f16_matrix_mad_k16(short8 a, int8 b, float8 acc);
```

**A operand** (`short8`): Each work-item (lane `lid`) holds `A[m_base+i][k_base+lid]` for i=0..7, packed as short8.

**B operand** (`int8`): VNNI format. Each lane `lid` holds pairs: `b.si = pack_half2(B[k_base+2i][n_base+lid], B[k_base+2i+1][n_base+lid])`.

**Result**: `acc[i] += dot(A_row_i[0:15], B_col_lid[0:15])` for i=0..7. Each lane computes one column of the 8×16 output tile.

---

## 5. 2D Block IO Functions Used

### Read A (32 rows × 16 cols)
```c
intel_sub_group_2d_block_read_16b_32r16x1c(
    base,           // global pointer (64B aligned)
    width_bytes,    // total row width in bytes (≥64)
    height,         // total number of rows
    pitch_bytes,    // row stride in bytes (≥width, multiple of 16)
    (int2)(x, y),   // x=column (element units), y=row
    dest);          // ushort[32] output per lane
```
Each lane gets one column of the 32×16 block. `dest[i]` = element at row `y+i`, column `x+lid`.

### Read B with VNNI Transform (16 rows × 32 cols)
```c
intel_sub_group_2d_block_read_transform_16b_16r16x2c(
    base, width_bytes, height, pitch_bytes,
    (int2)(x, y),   // reads 32 columns starting at x
    dest);          // uint[16] output per lane
```
Reads 16×32 and packs pairs of rows into uint (VNNI format). Output `dest[0..7]` = first 16 columns, `dest[8..15]` = next 16 columns. Directly usable as `int8` for DPAS B operand.

### Write C (8 rows × 16 cols)
```c
intel_sub_group_2d_block_write_16b_8r16x1c(
    base, width_bytes, height, pitch_bytes,
    (int2)(x, y),
    src);           // ushort[8] per lane
```

---

## 6. Build & Test Commands

```bash
# Build options for kernel
-cl-std=CL2.0 -cl-mad-enable -cl-fast-relaxed-math \
-Dcl_intel_subgroup_matrix_multiply_accumulate \
-Dcl_intel_subgroup_2d_block_io

# Compile host test
g++ -O2 -o test_gemm_f16 test_gemm_f16.cpp -lOpenCL

# Run
./test_gemm_f16
```

---

## 7. Dispatch Formula

```c
const int WG_M = 4, WG_N = 8, TILE_M = 32, TILE_N = 32, SG_SIZE = 16;
const int WG_TILE_M = WG_M * TILE_M;  // 128
const int WG_TILE_N = WG_N * TILE_N;  // 256

int grid_m = (M + WG_TILE_M - 1) / WG_TILE_M;
int grid_n = (N + WG_TILE_N - 1) / WG_TILE_N;

size_t local[2]  = { SG_SIZE * WG_M * WG_N, 1 };  // {512, 1}
size_t global[2] = { grid_m * local[0], grid_n };
```

---

## 8. Remaining Optimization Opportunities

| Idea | Expected Impact | Risk |
|------|----------------|------|
| Larger matrices (4096×4096) | Higher efficiency (more parallelism) | None |
| WG_M=8, WG_N=4 (still 512 thr) | Maybe: more A reuse | Negligible |
| K-unroll×2 inline (no extra buffer) | Maybe 1-2%: reduce loop overhead | Register pressure |
| Cooperative prefetch (all SGs prefetch next WG_TILE) | Unknown: L3 warmup | Message contention |
| Mixed precision accumulation tricks | Unlikely at this scale | Accuracy risk |

---

## 8.5 Breakthrough: WG Swizzle for L3 Locality (55% → 65%)

The largest single late-stage win came from **remapping which output tile each
WG processes** so that concurrently scheduled WGs share more L3 data.

**Problem**: Default OpenCL grid order interleaves N-direction WGs aggressively.
Concurrent WGs on the GPU end up touching different B columns, so each WG
brings B from DRAM independently → poor L3 reuse on B.

**Fix**: Walk M dimension in groups of `SWIZZLE` before incrementing N. With
`SWIZZLE=2`, every 2 consecutive WGs in linear order share the same B column
window → those 2 WGs reuse B from L3.

```c
// Inside the kernel, after computing default get_group_id(0/1):
const int grid_m = get_num_groups(0);
const int grid_n = get_num_groups(1);
const int linear = get_group_id(1) * grid_m + get_group_id(0);
const int block_size = SWIZZLE * grid_n;
const int block_id  = linear / block_size;
const int block_off = linear % block_size;
const int rem_m       = grid_m - block_id * SWIZZLE;
const int cur_swizzle = (rem_m < SWIZZLE) ? rem_m : SWIZZLE;
const int gn = block_off / cur_swizzle;
const int gm = block_id * SWIZZLE + (block_off % cur_swizzle);
// Then use gm, gn instead of get_group_id(0/1) for tile_m, tile_n.
```

**Tuning**: SWIZZLE=2 best on B580 (~65% K=2560, ~60% K=4096). SWIZZLE=4
worse on K=4096 (~59%). SWIZZLE=8 worst (~56% K=4096). Too-wide swizzle
overflows the L3 set that the scheduler can keep hot.

This is the same trick used by cuBLAS / CUTLASS / oneDNN ("L2 swizzle" /
"thread block swizzle"). On Xe2 the equivalent is L3 swizzle.

---

## 9. Key Takeaways

1. **2D block IO is mandatory** — It's the single biggest optimization (24% → 35%) and enables all subsequent gains by eliminating SLM overhead.

2. **Minimize messages per DPAS** — The ideal is 2 messages → 8 DPAS (ratio 4:1). Each additional message per k_step costs ~5% efficiency.

3. **512 threads per WG is the sweet spot on B580** — Balances L3 reuse (large WG) against occupancy (multiple WGs per XeCore).

4. **Don't fight the hardware scheduler** — Software pipelining, explicit prefetch, and manual latency hiding all hurt because the HW thread scheduler already handles this.

5. **Stay within 8 float8 accumulators** — 16 causes register spills. The register file (4KB/thread) is the hard constraint.

6. **DPAS confirms XMX usage** — Achieving 60+ TFLOPS (vs ~12 TFLOPS FPU-only) proves XMX units are engaged.

7. **K-unrolling has a sweet spot at ×2** — K-unroll×2 with separate A/B buffers (v22) and wide A read `32r16x2c` (v24) gets to ~57%. K-unroll×4 (v26) spills registers and drops to 29%.

8. **WG swizzle is the late-stage breakthrough** — Once compute/load pipeline is well-tuned, the next bottleneck is L3 bandwidth on B. Remapping WG order to keep concurrent WGs sharing B columns gives a clean **+7-10%** with no register cost.

9. **Wide 2D block reads cost no extra registers** — `32r16x2c` for A delivers 32×32 f16 (2 k-steps) in one message at the same per-lane footprint as two `32r16x1c` reads but lower latency.

10. **Some 2D block IO variants have layout quirks** — `intel_sub_group_2d_block_read_transform_16b_32r16x2c` compiles but packs data differently than `16r16x2c`; always validate with correctness check, not just timing.

11. **Prefetch builtins on Xe2 are fragile** — `intel_sub_group_2d_block_prefetch_*` returned `CL_INVALID_WORK_GROUP_SIZE (-54)` in our setup. The hardware scheduler already hides load latency well; prefetch was not needed once swizzle + wide reads were in place.

12. **Cooperative SLM load is *worse* than redundant L3 reads at WG_SIZE=512.** Two `barrier(CLK_LOCAL_MEM_FENCE)` per iteration at 512 threads/WG costs ~23 percentage points of efficiency on B580 — far more than the savings from eliminating 4× redundant B reads. OpenCL has no sub-WG sync primitive, so SLM cooperative loading requires WG barriers. To make this win you'd need either (a) WG_SIZE ≤ 128, which kills XeCore occupancy, or (b) double-buffered SLM amortizing barriers across many k-steps, which doubles SLM use and again forces occupancy down. **The L3 hierarchy on Xe2 with wide 2D block reads is already efficient enough that explicit cooperation does not pay back the synchronization cost.**

13. **`*1c` and `*2c` are the only column-count variants for 32-row 2D block reads on Xe2.** `intel_sub_group_2d_block_read_16b_32r16x4c` is not defined. Maximum A read width is 32×32 f16 per message.

14. **65% efficiency on K=2560 / 60% on K=4096 appears to be the practical OpenCL ceiling on B580.** Beyond this, reaching the 75-85% range likely requires moving to oneDNN's nGEN/asm ukernel approach (direct GRF allocation, scheduled DPAS-load interleaving below the OpenCL abstraction), or vendor-private intrinsics not exposed in OpenCL.
