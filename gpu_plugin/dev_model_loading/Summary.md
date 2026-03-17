# GPU Cache Loading Optimization: Approach Comparison

## Overview

Three approaches explore strategies for accelerating GPU model cache loading, targeting
Qwen3-30B-A3B (~14.5 GB blob) on a Windows machine with ~16 GB physical RAM.

**Approach A** and **Approach B** were earlier per-site explorations.
**Approach C** (implemented in `river/mmap_parallel_io_opt`) is the production implementation:
a pair of universal `std::streambuf` subclasses that inject parallel I/O transparently at the
stream layer, covering all plugins and both `enable_mmap` modes without per-plugin changes.

---

## Approach A — `river/mmap_parallel_io_opt`: mmap + Parallel memcpy

### Mechanism

The existing OpenVINO model-cache path memory-maps the cache blob (`MapViewOfFile`), exposing
it as an `ov::SharedStreamBuffer`-backed `std::istream`.  `BinaryInputBuffer::read()` detects
this stream type, extracts the raw pointer via `SharedStreamBuffer::get_ptr()`, and performs:

1. **`PrefetchVirtualMemory`** — issues concurrent NVMe reads at high queue depth to warm source
   pages before any `memcpy` thread touches them (avoiding per-thread VMM lock serialization on
   cold pages).
2. **`ov::parallel_for` memcpy** — splits the block into 2 MB chunks copied concurrently,
   saturating memory bandwidth from already-warm pages.
3. **`VirtualFree(MEM_RESET)`** *(fix added in this branch)* — releases consumed source pages
   from the process working set immediately after each block, preventing working-set accumulation.

### Key File

`src/plugins/intel_gpu/include/intel_gpu/graph/serialization/binary_buffer.hpp`
— `BinaryInputBuffer::read()` hot path.

### Memory Pressure Model

| Component | Ownership | Eviction policy |
|---|---|---|
| Source (mmap view) | **Process working set** | Only via explicit `MEM_RESET` / `VirtualFree` |
| Destination (GPU alloc) | Process committed pages | OS can page-file swap under pressure |

Without `MEM_RESET`, every 96 MB block leaves its source pages in the process working set.
After ~150 blocks the cumulative source footprint reaches ~14 GB.  Combined with ~14 GB of
growing destination allocations, physical RAM is exhausted and Windows begins evicting
recently-prefetched pages from the pagefile before `memcpy` can consume them — causing
500–2900 ms stalls (read throughput drops from ~8 GB/s to <0.1 GB/s).

With `MEM_RESET` applied, source working-set peaks at ~1 block (~96 MB) rather than
accumulating, reducing init time from 17.86 s → 13.24 s.  However, destination-side
accumulation (GPU alloc, ~12.7 GB at the final stall point) still triggers pagefile access,
so two 1400–1675 ms stalls remain near the end of loading.

### Root Limitation

`VirtualFree(MEM_RESET)` on a `MapViewOfFile`-backed section is not guaranteed to succeed
(MSDN: it is valid only for private committed pages; file-mapped section pages may silently
fail).  Even when it succeeds, the destination pressure is outside the reach of this layer.

---

## Approach B — `river/cache_loading_opt`: Direct Parallel `ReadFile`

### Mechanism

`data.hpp` `load_weights()` is extended with an optional `weights_path` parameter.  When the
path is available and `data_size >= 4 MB`, it bypasses the `std::istream` entirely and calls
`ov::util::read_binary_file_parallel()` (`src/common/util/src/file_util.cpp`):

1. Each worker thread opens its own `HANDLE` via `CreateFileW(..., FILE_ATTRIBUTE_NORMAL)`.
2. Threads issue `ReadFile` with an `OVERLAPPED` struct carrying the absolute file offset,
   splitting the block across `min(hardware_concurrency, size / 1 MB)` threads.
3. Data flows directly: **NVMe → (OS kernel file cache) → destination buffer** without any
   intermediate memory-mapped source view in the process working set.

For the `is_alloc_host_accessible = false` (discrete GPU) path, the intermediate staging
buffers are upgraded from `std::vector<uint8_t>` (pageable, triggers hidden pinned-memory
bounce copies) to `allocation_type::usm_host` (directly page-locked, zero-copy DMA).

### Memory Pressure Model

| Component | Ownership | Eviction policy |
|---|---|---|
| Source (kernel file cache) | **OS owned** (soft pages) | OS evicts freely under any pressure |
| Destination (GPU alloc) | Process committed pages | OS can page-file swap under pressure |

The process never holds "source" pages in its own working set.  Total process RAM pressure is
half that of Approach A: only destination (~14.5 GB) instead of source + destination (both
~14.5 GB).

---

## Comparison Table

| Dimension | Approach A (mmap + parallel memcpy) | Approach B (direct parallel ReadFile) |
|---|---|---|
| **Source page ownership** | Process working set (problematic) | OS kernel cache (transparent) |
| **Peak process RAM (16 GB machine)** | source + dest ≈ 29 GB → pagefile thrash | dest only ≈ 14.5 GB → borderline |
| **NVMe queue depth** | `PrefetchVirtualMemory` (advisory, 1 call) | N concurrent `ReadFile` handles (guaranteed) |
| **MEM_RESET reliability** | May silently fail on file-mapped views | Not needed |
| **Offset compensation** | Not needed (mmap view is already offset-aware) | Required: physical file offset ≠ logical stream offset |
| **Works without `weights_path`** | Yes (uses existing mmap stream) | No (fallback to `sgetn`) |
| **Code change scope** | `binary_buffer.hpp` only | `data.hpp` + `file_util.cpp` + `file_util.hpp` |
| **Init time (Qwen3-30B, ~16 GB RAM)** | 13.24 s (with MEM_RESET fix) | 7.77 s |
| **Init time baseline (no opt)** | ~20–22 s | ~20–22 s |

---

## Observed Benchmark Data (Approach A, with MEM_RESET fix)

- **Run 1 (no fix):** 12.85 s — throughput degrades from 8.6 → 5–7 GB/s after 6 GB; outliers at 48 ms, 970 ms, 1370 ms.
- **Run 2 (no fix, different run):** 17.86 s — 2902 ms stall at offset 12.4 GB, 514 ms at 12.7 GB.
- **Run 3 (MEM_RESET fix):** 13.24 s — two remaining stalls: 1447 ms at offset 12.7 GB, 1675 ms at 13.2 GB. PrefetchVirtualMemory drops from 15–17 ms → 8–10 ms for blocks 60–100 (source pages served from OS cache freed by MEM_RESET), confirming the fix works for source pressure; stalls shift entirely to destination pressure.

---

## Benchmark: Per-Block Latency Comparison (Qwen3-30B-A3B, Windows, ~16 GB RAM)

**Test conditions:** 144 × 96 MB blocks for each approach. mmap timings combine
`PrefetchVirtualMemory` + `parallel memcpy` into a single end-to-end latency per block
(the full wall-clock cost visible to the caller). `file_read_parallel` timings are measured
internally from open→read→close.

### Summary Table

| Metric | Approach A: mmap (Prefetch+memcpy) | Approach B: file_read_parallel |
|---|---|---|
| Block count (96 MB) | 144 | 144 |
| **Avg latency** | **49.6 ms** | **22.6 ms** |
| **Avg effective BW** | **1.89 GB/s** | **4.15 GB/s** |
| p50 latency | 28.2 ms | 22.5 ms |
| p50 effective BW | 3.32 GB/s | 4.16 GB/s |
| p95 latency | 34.9 ms | 37.0 ms |
| Best block BW | 4.61 GB/s | **10.24 GB/s** |
| Worst block latency | **1693 ms** | 49 ms |
| Worst block BW | 0.055 GB/s | 1.93 GB/s |
| Stall outliers (>100 ms) | **3** | **0** |

### Approach A: Internal Time Breakdown (normal blocks, stalls excluded)

Excluding the 3 stall blocks (>100 ms), the remaining 141 blocks average **27.5 ms = 3.41 GB/s**:

| Phase | Avg time | Share |
|---|---|---|
| `PrefetchVirtualMemory` | 14.7 ms | **53%** |
| `parallel memcpy` | 12.8 ms | 47% |

Over half of each block's wall time is spent in `PrefetchVirtualMemory`, which has no equivalent
cost in Approach B. The `parallel memcpy` phase itself (12.8 ms) is very close to Approach B's
per-block read time, confirming that the warm-page copy speed of both methods is comparable —
the gap comes entirely from the mandatory prefetch overhead and occasional pagefile stalls.

### Approach B: Cold vs. Warm Phase

`read_binary_file_parallel` shows no meaningful difference between cold and warm phases:

| Phase | Blocks | Avg latency | Avg BW |
|---|---|---|---|
| Cold (first 63 blocks, OS cache empty) | 63 | 22.7 ms | 4.12 GB/s |
| Warm (next 81 blocks, OS cache filling) | 81 | 22.5 ms | 4.17 GB/s |

This confirms that concurrent `ReadFile` handles saturate NVMe queue depth independently of OS
page cache state, providing stable throughput from the very first block.

### Key Takeaways

1. **Approach B is 2.2× faster on average** (4.15 vs 1.89 GB/s) when stalls are included.
2. **Approach B is ~20% faster in the normal case** (3.41 GB/s without stalls vs 4.15 GB/s),
   with the entire gap attributable to the `PrefetchVirtualMemory` overhead in Approach A.
3. **Approach A has catastrophic tail latency** (up to 1693 ms, 0.055 GB/s) caused by pagefile
   re-faults when destination GPU allocations exhaust physical RAM. Approach B has no such
   outliers (worst block: 49 ms, 1.93 GB/s).
4. **Approach A's `parallel memcpy` phase alone** (12.8 ms, ~7.5 GB/s) is faster than
   Approach B's single-file read (22.5 ms) — the mmap strategy would win if `PrefetchVirtualMemory`
   could be eliminated and working-set pressure did not exist.

---

## Recommendation

For machines where `weights_path` is available (the common production path), **Approach B is
architecturally superior**: it eliminates the source-side working-set problem at the root by
never creating a process-owned source mapping, and provides NVMe queue depth through independent
file handles rather than advisory prefetch calls.

**Approach A** remains valuable as a fallback for cases without a path (e.g., in-memory streams
or encrypted caches) and requires no changes outside `binary_buffer.hpp`.

The ideal production implementation combines both:
- Use **Approach B** (`read_binary_file_parallel`) when `weights_path` is available.
- Fall back to **Approach A** (mmap + `MEM_RESET`) when only a stream is available and the
  stream is `SharedStreamBuffer`-backed.
- Fall back to standard `sgetn` otherwise.

> Approach C (below) supersedes this recommendation by implementing both strategies in a single
> universal layer, removing the need for per-plugin code changes entirely.

---

## Approach C — `river/mmap_parallel_io_opt`: Universal Parallel I/O via `std::streambuf`

### Design Philosophy

Rather than patching each plugin's deserialization site, inject parallel I/O at the `std::streambuf`
layer. Any `std::istream&` consumer — `BinaryInputBuffer`, all plugin deserializers, encrypted
cache wrappers — gets the speedup transparently.

Two complementary class implementations live in `src/common/util/include/openvino/util/`:

| Class | Backing | Platform |
|---|---|---|
| `ParallelReadStreamBuf` | Direct file I/O (`pread` / `OVERLAPPED ReadFile`) | Linux + Windows |
| `ParallelMemStreamBuf` | In-memory region (mmap-backed tensor) with auto file-detection | Linux + Windows |

### `ParallelReadStreamBuf`

Overrides `xsgetn()`: for reads ≥ 4 MB, splits the request across `N = min(hardware_concurrency, size / 1 MB)`
threads using **absolute-offset** reads (`pread` on Linux, `OVERLAPPED ReadFile` on Windows).
Each thread owns its own native handle opened in `FILE_SHARE_READ` mode, so there is no shared
seek pointer and no per-read lock.

**Key fixes over the Approach B sketch:**
- `underflow()` uses an 8 KB internal buffer so that single-character reads (e.g. `std::getline`
  in `CompiledBlobHeader`) work correctly without an infinite loop.
- `seekoff()`/`seekpos()` update `m_file_offset` so the streambuf is seek-compatible with all
  consumers.
- `showmanyc()` returns remaining byte count so `std::istream::read` avoids spurious `underflow`
  calls on large sequential reads.

### `ParallelMemStreamBuf`

Wraps an in-memory region (typically an mmap-backed `ov::Tensor`). For large reads, uses
`ov::parallel_for` + `memcpy` — but more importantly:

#### File-backed mmap auto-detection and delegation

When constructed, `ParallelMemStreamBuf` checks whether the pointer actually belongs to a
**file-backed** mapping:

- **Windows**: `VirtualQuery()` → `MEM_MAPPED` type check → `GetMappedFileNameW()` → `QueryDosDevice()`
  converts the kernel device path (`\Device\HarddiskVolume3\...`) to a Win32 path (`C:\...`).
- **Linux**: parses `/proc/self/maps` to find the mapping's file path and `map_offset`.

If the mapping is file-backed **and** `size >= threshold` (4 MB), `ParallelMemStreamBuf`
immediately creates an internal `ParallelReadStreamBuf` over the same file at the computed
file offset, and **delegates all virtual method calls** (`xsgetn`, `underflow`, `uflow`,
`seekoff`, `seekpos`, `showmanyc`) to it.

This means `ENABLE_MMAP=True` silently becomes a direct `ReadFile`/`pread` path — the mmap
view is never read from at all — eliminating the 2× working-set pressure that caused the stalls
in Approach A.

For non-file-backed memory (anonymous mmap, USM host buffers), fallback to
`PrefetchVirtualMemory` (Windows) / `madvise(MADV_WILLNEED)` (Linux) + parallel `memcpy`.

### Integration Points (no per-plugin changes required)

| Code path | Change |
|---|---|
| `Plugin::import_model(Tensor)` path (`enable_mmap=true`) | Wrap tensor pointer in `ParallelMemStreamBuf` |
| `FileStoreCacheManager::read_cache_entry` (`enable_mmap=false`) | Replace `std::ifstream` with `ParallelReadStreamBuf` |
| `SingleFileStorage::get_cached_model` (`enable_mmap=false`) | Already uses `ParallelReadStreamBuf` |
| IR frontend `.bin` read (`compile_model`) | Replace `std::ifstream` with `ParallelReadStreamBuf` |
| GPU `data_inst::attach_or_copy_data` (weight Host→GPU copy) | `ov::parallel_for` + `memcpy` for tensors ≥ 4 MB |
| Linux mmap object (`lin_mmap_object.cpp`) | `madvise(MADV_SEQUENTIAL | MADV_WILLNEED)` after `mmap()` |
| Windows mmap object (`win_mmap_object.cpp`) | `PrefetchVirtualMemory` after `MapViewOfFile` |

---

## Approach C Performance Data

### Test Conditions

- Model: Qwen3-30B-A3B (FP16-4BIT, ~14.5 GB blob), `ENABLE_MMAP=True`
- Platform: Windows, NVMe SSD, ~16 GB physical RAM
- **OLD** = `ParallelMemStreamBuf` mmap + parallel `memcpy` (48 threads, no file-detection)
- **OPT** = `ParallelMemStreamBuf` detects file-backed mmap → delegates to `ParallelReadStreamBuf` (16 threads ReadFile)
- Runtime: `2026.1.0-21317-ffb995e48e4-river/mmap_parallel_io_opt`

### End-to-End Pipeline Initialization Time

| | OLD (mmap+memcpy) | OPT (ReadFile delegation) | Δ |
|---|---|---|---|
| **Pipeline init time** | **13.48 s** | **8.61 s** | **−4.87 s (−36%)** |
| Total IO time (Σ chunk) | 5.30 s | 4.77 s | −0.53 s |
| **Non-IO overhead** | **8.18 s** | **3.84 s** | **−4.34 s (−53%)** |
| Effective IO bandwidth | 2.737 GB/s | 3.041 GB/s | +11% |
| Total data read | 14.50 GB | 14.50 GB | — |

The bulk of the saving (4.34 s / 89% of total) is in **non-IO overhead** — the OS working-set
management cost of maintaining a 13.5 GB mmap view simultaneously with a 13.5 GB destination
allocation, which was never visible in per-chunk timing.

### Per-Chunk Bandwidth Distribution (96 MB chunks, n = 144)

| Metric | OLD (mmap+memcpy) | OPT (ReadFile) |
|---|---|---|
| **Average** | **5.297 GB/s** | 3.105 GB/s |
| **Median** | **6.383 GB/s** | 2.900 GB/s |
| **Std-dev** | **2.523** | **0.749** (3.4× more stable) |
| p10 | 1.982 GB/s | 2.365 GB/s |
| p25 | 2.692 GB/s | 2.512 GB/s |
| p75 | 7.424 GB/s | 3.690 GB/s |
| p90 | 8.215 GB/s | 3.955 GB/s |
| p99 | 8.756 GB/s | 4.985 GB/s |
| **Min** | **0.070 GB/s** | **2.101 GB/s** |
| Max | 8.992 GB/s | 8.816 GB/s |
| Hard stalls (< 1 GB/s) | **3** | **0** |
| Below 2 GB/s | **26** | **0** |

### Why OLD Average Looks Higher but Wall Time is Worse

The per-chunk timer in `ParallelMemStreamBuf::parallel_copy` measures only the `memcpy` phase
**after** the pages are resident. When `PrefetchVirtualMemory` has warmed pages from DRAM, the
`memcpy` is fast (7–9 GB/s) and makes the average look good. The real cost is:

1. **Hidden OS overhead** (4.34 s): `PfnLock` contention for 3.4 M page-frame-number entries,
   Working Set Manager trim events, TLB shootdown — none of which appear in the chunk timer.
2. **26 chunks below 2 GB/s**: early Working Set pressure degrades throughput even in
   non-stall chunks (OLD p10 = 1.98 GB/s vs OPT p10 = 2.37 GB/s).
3. **3 Hard Page Fault stalls** (min = 0.070 GB/s → single-chunk latency ≈ 1.37 s): Windows
   Working Set Manager evicts mmap source pages to Standby before a `memcpy` thread can consume
   them; the thread re-faults them from NVMe, costing 500–1400 ms.

The OPT path has none of these: `ReadFile` writes directly into the caller's buffer without
creating any source-side working-set residency. The process always holds only the destination
buffer (~14.5 GB), half the peak RAM of the old path.

### Comparison Across All Three Approaches

| Approach | Mechanism | Pipeline init | Notes |
|---|---|---|---|
| A (baseline) — no fix | mmap + `PrefetchVirtualMemory` + `parallel memcpy`, no `MEM_RESET` | ~17.9 s | Worst-case stall 2.9 s |
| A (with MEM_RESET fix) | Same + `VirtualFree(MEM_RESET)` after each block | 13.24 s | Destination pressure still causes 2 residual stalls |
| B — direct ReadFile per site | `read_binary_file_parallel` in `data.hpp` GPU plugin only | 7.77 s | Requires `weights_path`; GPU plugin only |
| **C — universal streambuf** | `ParallelMemStreamBuf` auto-detects file-backing → `ParallelReadStreamBuf` | **8.61 s** | All plugins, both mmap modes, cross-platform |

Approach C is within 11% of Approach B's wall time while eliminating the need for per-plugin
`weights_path` threading and working across all plugins, both `enable_mmap` modes, and on both
Windows and Linux.

### Remaining Overhead Analysis (OPT: 8.61 s total)

| Component | Estimated time |
|---|---|
| ReadFile I/O (Σ chunks) | 4.77 s |
| GPU kernel compilation / weight upload | ~3.50 s (unchanged, non-IO) |
| Misc deserialization overhead | ~0.34 s |

The non-IO portion (3.84 s) is now dominated by GPU compilation time rather than OS memory
management, meaning further improvement requires GPU-side parallelism, not I/O changes.
