# PTL 4Xe iGPU — Hardware Probe (2026-05-08)

**Target**: `intel@10.239.152.140` — Intel PTL 4Xe Linux iGPU
**Driver**: NEO 26.14.037858, OpenCL 3.0
**Method**: custom OpenCL probes in `utils/hw_probe/` (see `run_ptl_4xe.sh`)

## Spec parity check (clinfo + gpu_info.c)

| Field | Reported | SKILL §3 spec |
|---|---:|---:|
| Compute units (Xe × EU) | 32 | 4 × 8 = 32 ✓ |
| Max clock | 2450 MHz | 2400/2450 MHz ✓ |
| Subgroup sizes | 16, 32 | 16, 32 ✓ |
| SLM (per Xe core) | 128 KiB | 32 KiB per workgroup (per-Xe pool is 128 KiB) ✓ |
| L3 cache | 4 MiB | (small — bandwidth tests confirm) |
| Global memory | 21.2 GiB | shared with system DRAM ✓ |
| Cacheline | 256 B | matches Xe2 |

Threads/EU = 10 is not exposed via OpenCL; matches PTL Xe3 spec from SKILL.

## Memory bandwidth (custom streaming copy, L2-defeating rotation)

| Buffer × 3 | Iters | Copy R+W (GB/s) | Read-only (GB/s) |
|---:|---:|---:|---:|
|  512 MiB | 30 | **103.93** | 98.29 |
| 1024 MiB | 20 | **103.85** | 97.57 |
| 2048 MiB | 10 | 97.06 | 84.06 |

**Sustained achievable BW ≈ 100 GB/s** (R+W) for working sets that fit dual-channel LPDDR5X but exceed the 4 MiB L3.

Theoretical peak (LPDDR5X-7467 × 128-bit): **110 GB/s** → measured ≈ **94%** of theoretical at 1 GiB; 88% at 2 GiB (likely page-walk / TLB pressure).

> For roofline purposes, use **BW = 100 GB/s** as the achievable ceiling on PTL 4Xe (slightly more conservative than the nameplate 110 GB/s used in the qwen3_omni 4Xe SUMMARY, which corresponds to vendor-spec).

## Theoretical compute peaks (recomputed)

Using SKILL §5 formula: `XMX_peak = N_Xe × EU/Xe × FLOPs_per_cycle × clock`
- 4 Xe × 8 EU = 32 EU at 2.45 GHz
- FP16 XMX: 32 × 256 FLOPs/cycle × 2.45 GHz = **20.07 TFLOPS**
- INT8 XMX: 2× = **40.14 TOPS**

Ridge point (FP16, achieved): 20070 / 100 = **~200 FLOP/byte**.

## Files
- Probes: [gpu_info.c](.github/skills/dev_roofline_profiling/utils/hw_probe/gpu_info.c), [mem_bw.c](.github/skills/dev_roofline_profiling/utils/hw_probe/mem_bw.c)
- Driver script: [run_ptl_4xe.sh](.github/skills/dev_roofline_profiling/utils/hw_probe/run_ptl_4xe.sh)

## Reproduction
```bash
.github/skills/dev_roofline_profiling/utils/hw_probe/run_ptl_4xe.sh
```
