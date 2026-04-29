# Windows A/B Test: ParallelReadStreamBuf vs std::ifstream

## Test Environment

- **OS**: Windows (Visual Studio 17 2022, MSBuild)
- **Device**: GPU (Intel)
- **Storage**: NVMe SSD
- **Model**: qwen3-30b-a3b (`.bin` weights: 16.3 GB, cache blob: ~15.4 GB)
- **Branch**: `river/mmap_parallel_io_opt` (PR #34679)
- **Build**: Release, parallel 16
- **Test tool**: `llm_bench/benchmark.py -d GPU -n 0 -ic 32 -b 1 -mc 1`

## Files Changed (4 call sites)

| File | Baseline (master) | Optimized (branch) |
|---|---|---|
| `src/frontends/ir/src/frontend.cpp` | `std::ifstream` for `.bin` weights | `ParallelReadStreamBuf` |
| `src/inference/src/cache_manager.hpp` | `std::ifstream` for cache blob | `ParallelReadStreamBuf` |
| `src/inference/src/single_file_storage.cpp` | `std::ifstream` + `seekg` for cache blob | `ParallelReadStreamBuf` with offset |
| `src/plugins/intel_gpu/src/plugin/plugin.cpp` | `SharedStreamBuffer` for mmap Tensor import | `ParallelMemStreamBuf` (detects mmap, falls back to file-based parallel read) |

## Test Methodology

- Source files swapped via `copy /Y`, then **PowerShell timestamp touch** to force MSBuild recompilation
- Each scenario run 3 times sequentially
- Run 1 benefits from warm page cache (prior build/install touched large files); Runs 2-3 reflect cold/steady-state cache behavior
- DLLs deployed to both `release_install/` and `py310/Lib/site-packages/openvino/libs/`
- `[DIAG]` timing instrumentation added to all 4 files + `core_impl.cpp` + `parallel_read_streambuf.cpp`

## Results

### Scenario 1: ENABLE_MMAP=false (read .bin weights + read cache blob via ifstream)

#### read_model — openvino_model.bin (16.3 GB)

| Run | Baseline (ifstream) | Optimized (ParallelReadStreamBuf) | Speedup |
|---|---|---|---|
| 1 (warm cache) | 23,873 ms | 19,275 ms | 1.2x |
| 2 (cold cache) | 217,394 ms | 12,982 ms | **16.7x** |
| 3 (cold cache) | 218,121 ms | 12,020 ms | **18.1x** |

#### cache_manager::read_cache_entry — cache blob (~15.4 GB)

| Run | Baseline (ifstream) | Optimized (ParallelReadStreamBuf) | Speedup |
|---|---|---|---|
| 1 | 42,267 ms | 18,749 ms | **2.3x** |
| 2 | 38,179 ms | 17,281 ms | **2.2x** |
| 3 | 38,760 ms | 18,894 ms | **2.1x** |

#### Pipeline initialization time (end-to-end)

| Run | Baseline | Optimized | Speedup |
|---|---|---|---|
| 1 (warm cache) | 67.54s | 40.04s | 1.7x |
| 2 (cold cache) | 257.12s | 32.34s | **7.9x** |
| 3 (cold cache) | 258.41s | 32.87s | **7.9x** |

### Scenario 2: ENABLE_MMAP=true (mmap weights + cache blob via Tensor import_model)

#### cache_manager::read_cache_entry — cache blob (~15.4 GB)

| Run | Baseline (SharedStreamBuffer) | Optimized (ParallelMemStreamBuf) | Speedup |
|---|---|---|---|
| 1 | 17,842 ms | 6,082 ms | **2.9x** |
| 2 | 17,950 ms | 6,212 ms | **2.9x** |
| 3 | 19,506 ms | 6,361 ms | **3.1x** |

#### Pipeline initialization time (end-to-end)

| Run | Baseline | Optimized | Speedup |
|---|---|---|---|
| 1 | 18.88s | 7.15s | **2.6x** |
| 2 | 18.97s | 6.86s | **2.8x** |
| 3 | 20.39s | 7.08s | **2.9x** |

## Key Findings

1. **Cold cache mmap=false shows the largest improvement**: `read_model` drops from ~218s to ~12s (**18x**). `ParallelReadStreamBuf` uses `ReadFile` + `FILE_FLAG_NO_BUFFERING` to bypass Windows page cache contention that cripples `std::ifstream` when reading large files under memory pressure.

2. **Cache blob read (mmap=false)**: Consistent ~2.2x improvement (39s → 18s). The parallel prefetch + large sequential reads outperform single-threaded `ifstream`.

3. **Pipeline total time (mmap=false, cold)**: From ~258s to ~33s (**7.9x**). The combined effect of faster weights loading and faster cache import.

4. **mmap=true also benefits significantly**: `ParallelMemStreamBuf` detects mmap-backed memory and creates a `ParallelReadStreamBuf` to read directly from the file, bypassing the kernel's mmap page fault path. Cache import drops from ~18s to ~6s (**2.9x**), and Pipeline total from ~19s to ~7s (**2.8x**).

5. **Warm cache mmap=false (Run 1)**: Modest 1.2x-1.7x improvement — when data is already in page cache, the bottleneck shifts from I/O to memory copy, where `ParallelReadStreamBuf` still helps via larger buffer reads but the margin is smaller.

## Root Cause: Why ifstream is Slow on Windows

When multiple large files (16+ GB each) are read sequentially with `std::ifstream`, Windows page cache becomes saturated. The OS evicts pages from the first file to make room for the second, causing **page cache thrashing**. Subsequent reads of either file trigger cold disk I/O at ~75 MB/s effective throughput instead of memory-speed access.

`ParallelReadStreamBuf` avoids this by:
- Using `FILE_FLAG_NO_BUFFERING` — bypasses the page cache entirely, reading directly from disk to user buffer
- Using 8 MB read-ahead with `posix_fadvise`/`ReadFile` prefetch — keeps the disk pipeline full
- Achieving ~1.3 GB/s effective throughput on NVMe vs ~75 MB/s for thrashing ifstream

## MSBuild Incremental Build Caveat

During testing, we discovered that Windows `copy /Y` does not always trigger MSBuild recompilation — the build system may reuse stale `.obj` files. The fix is to touch file timestamps after copying:

```bat
powershell -Command "(Get-Item 'path\to\source.cpp').LastWriteTime = Get-Date"
```

This was critical for valid A/B comparison and invalidated two earlier test attempts before being identified.
