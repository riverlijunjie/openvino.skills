---
name: model-loading-optimization
description: 'Optimization and debug guide for OpenVINO model loading I/O performance. Use when diagnosing slow model/cache loading, optimizing parallel I/O paths, fixing mmap memory pressure, or understanding the ParallelReadStreamBuf / ParallelMemStreamBuf architecture. Covers both enable_mmap=true and enable_mmap=false paths on Linux and Windows.'
---

# OpenVINO Model Loading Optimization

Use this skill when:
- Model or cache blob loading is slower than expected (target: saturate NVMe bandwidth).
- Peak memory is excessive during model loading (mmap double-occupancy problem).
- You need to understand which I/O path is active for a given `enable_mmap` setting.
- Adding parallel I/O to a new call site (new plugin, new frontend, new cache format).
- Debugging data corruption from incorrect file offset compensation.
- Diagnosing Linux page cache pressure or Windows working set thrashing during large model loads.

Do not use this skill when:
- The issue is GPU kernel compilation time (JIT), not I/O.
- The bottleneck is network download, not local disk read.
- The model is small enough (< 4 MB) that parallel I/O overhead exceeds the benefit.

## What This Skill Helps Answer

- Which I/O class is used for a given `enable_mmap` and plugin combination?
- Why does mmap cause 2× physical RAM usage and how does `ParallelMemStreamBuf` eliminate it?
- Why must each `pread` thread have its own file descriptor on Linux?
- Where does `std::ifstream` bottleneck on large files and what replaces it?
- How does file-backed mmap auto-detection work on Linux vs Windows?
- Which call sites have been optimized and which still use `std::ifstream`?

## Mental Model

Model loading I/O in OpenVINO has two independent dimensions:

1. **mmap vs non-mmap**: Controlled by `enable_mmap` config + plugin `caching_with_mmap` capability.
2. **Cache hit vs cache miss**: Parallel I/O classes activate **only on cache-hit paths**. Cache miss does JIT compilation — no file read via these classes.

The core optimization replaces all large sequential reads with parallel positional I/O at the `std::streambuf` layer:

- `ParallelReadStreamBuf` — wraps a file path; uses multi-thread `pread`/`ReadFile` for reads ≥ 4 MB.
- `ParallelMemStreamBuf` — wraps a memory pointer (mmap tensor); auto-detects file-backed mmap and delegates to `ParallelReadStreamBuf`, avoiding page faults entirely.

Because the optimization lives at the streambuf level, any `std::istream&` consumer gets the speedup transparently — no per-plugin changes required.

The key memory insight: mmap causes both source pages (file cache) AND destination pages (weight buffers) to reside in physical RAM simultaneously (2× blob size). `ParallelMemStreamBuf` eliminates source-page residency by reading via `pread` instead of faulting mmap pages, cutting peak RAM to 1× blob size.

## Model Loading Code Path Map

### 1. Config plumbing — determining which I/O path is active

Start here when the question is "which path is my model actually taking?"

Look at:
- `src/inference/src/dev/core_impl.cpp` (~line 1565)
- `src/inference/src/cache_manager.hpp` (~line 62, 71)

Look for:
- `m_mmap_enabled` — propagated from user config via `CoreConfig` → `CacheContent`
- `device_supports_internal_property(plugin, ov::internal::caching_with_mmap)` — gate for mmap path
- `std::visit(model_importer, compiled_blob)` — variant dispatch: index 0 = `ov::Tensor` (mmap), index 1 = `std::istream&` (stream)
- `ov::read_tensor_data(path, blob_offset)` — creates mmap-backed tensor

Questions to answer:
- Is the user config `ENABLE_MMAP` actually reaching the loading code?
- Does the target plugin advertise `caching_with_mmap`?
- Is this a cache hit or cache miss? (cache miss never reaches parallel I/O classes)

### 2. Non-mmap path — `ParallelReadStreamBuf` direct file I/O

Start here when `enable_mmap=false` or the plugin does not support mmap.

Look at:
- `src/common/util/include/openvino/util/parallel_read_streambuf.hpp` — declaration
- `src/common/util/src/parallel_read_streambuf.cpp` — all implementations
- `src/inference/src/cache_manager.hpp:71` — cache blob read
- `src/inference/src/single_file_storage.cpp:280` — single-file cache read with byte offset
- `src/frontends/ir/src/frontend.cpp` — IR `.bin` weights read (enable_mmap=false branch)

Look for:
- `xsgetn()` override — threshold check (4 MB), then `parallel_read()` or `single_read()`
- Per-thread fd open: `open(path, O_RDONLY | O_CLOEXEC)` (Linux) / `CreateFileW(...)` (Windows)
- Thread count: `min(hardware_concurrency, size / 1MB)`
- 4 KiB chunk alignment: `(chunk_size + 4095u) & ~size_t{4095u}`
- `underflow()` 8 KB internal buffer for single-char reads
- `seekoff()` / `seekpos()` logical-vs-absolute offset conversion

Platform-specific I/O:
- **Linux**: `pread(fd, dst, size, offset)` — thread-safe positional read, no seek pointer
- **Linux optimization**: `posix_fadvise(WILLNEED)` before read, `posix_fadvise(DONTNEED)` after — controls page cache pressure
- **Windows**: `SetFilePointerEx` + `ReadFile` with per-thread `HANDLE`

Questions to answer:
- Is the read large enough (≥ 4 MB) to trigger parallel I/O?
- Are per-thread file descriptors being used? (sharing fd kills Linux readahead — see Traps)
- Is the file offset correct? (header offset compensation)

### 3. Mmap path — `ParallelMemStreamBuf` with auto file-detection

Start here when `enable_mmap=true` and the plugin supports `caching_with_mmap`.

Look at:
- `src/plugins/intel_gpu/src/graph/common_utils/parallel_mem_streambuf.hpp` — declaration
- `src/plugins/intel_gpu/src/graph/common_utils/parallel_mem_streambuf.cpp` — implementations + platform detection
- `src/plugins/intel_gpu/src/plugin/plugin.cpp:460` — GPU plugin creates `ParallelMemStreamBuf`
- `src/plugins/intel_gpu/src/plugin/plugin.cpp:760` — GPU plugin advertises `caching_with_mmap`

Look for:
- Constructor: file-backed mmap detection
  - **Linux**: parses `/proc/self/maps` to match address → file path + offset
  - **Windows**: `VirtualQuery()` → `MEM_MAPPED` check → `GetMappedFileNameW()` → `resolve_device_path()` converts kernel path to Win32 path
- `m_file_buf` — if file-backed, creates internal `ParallelReadStreamBuf` and delegates ALL virtual methods
- Fallback path: `madvise(MADV_WILLNEED)` (Linux) / `PrefetchVirtualMemory` (Windows) + `parallel_for(memcpy)`
- Windows fallback caps at 16 concurrent chunks (PFN database lock contention)

Questions to answer:
- Is the mmap region file-backed or anonymous? (anonymous = USM host / in-memory blob → fallback path)
- Is `m_file_buf` non-null? If yes, all reads bypass mmap page faults entirely.
- Is the file offset computation correct? (`map_file_offset + (addr - range_start)`)

### 4. GPU weight loading — Host→GPU copy optimization

Start here when optimizing discrete GPU weight transfer after file I/O completes.

Look at:
- `src/plugins/intel_gpu/src/graph/data.cpp` — `attach_or_copy_data` weight loading

Look for:
- `ov::parallel_for` + `memcpy` for tensors ≥ 4 MB (Host→Host copy before GPU DMA)
- USM host allocation (`allocation_type::usm_host`) instead of `std::vector<uint8_t>` for staging buffers
- Double-buffering for discrete GPU: ping-pong between two USM buffers overlapping CPU read and GPU DMA

Questions to answer:
- Is the staging buffer USM host-allocated? (pageable `std::vector` causes hidden pinned-memory bounce copies)
- Is the chunk size L3-cache-friendly (~4 MB)? Too large → L3 eviction before GPU DMA can snoop.

### 5. IR frontend weights read

Start here when `read_model` (`.bin` file loading) is the bottleneck rather than cache import.

Look at:
- `src/frontends/ir/src/frontend.cpp` — `load_impl()` weights loading path

Look for:
- `enable_mmap=false` branch: should use `ParallelReadStreamBuf` instead of `std::ifstream`
- `enable_mmap=true` branch: mmap path (no change needed)

Questions to answer:
- Is the `.bin` file being read with `std::ifstream`? That's ~75 MB/s on Windows for 16+ GB files.
- Has `ParallelReadStreamBuf` been applied here? (This was the root cause of a 220s → 9s improvement)

### 6. Unit tests

Look at:
- `src/inference/tests/unit/parallel_read_streambuf_test.cpp`
- `src/plugins/intel_gpu/tests/unit/test_utils/parallel_mem_streambuf_test.cpp`

Build and run:
```bash
cd build-x86_64-release
make ov_inference_unit_tests -j$(nproc)
./bin/intel64/Release/ov_inference_unit_tests --gtest_filter=*ParallelReadStreamBuf*
./bin/intel64/Release/ov_inference_unit_tests --gtest_filter=*ParallelMemStreamBuf*
./bin/intel64/Release/ov_inference_unit_tests --gtest_filter=*SingleFileStorageTest*
```

## Symptom To First Inspection Point

- Model loading very slow (< 100 MB/s) on NVMe-capable hardware:
  Check if `std::ifstream` is still used at the call site (Section 2, 5). Large files need `ParallelReadStreamBuf`.

- Near-OOM or excessive memory during loading with `enable_mmap=true`:
  Check Section 3. Verify `ParallelMemStreamBuf` is detecting file-backed mmap and delegating to `ParallelReadStreamBuf` (bypasses mmap page faults, saves ~1× blob size in RAM).

- Loading fast first time but slow on subsequent runs:
  Check page cache pressure. Linux: verify `posix_fadvise(DONTNEED)` is releasing pages after read. Windows: verify mmap pages aren't accumulating in working set.

- Data corruption or garbled weights:
  Check file offset compensation (Section 2). Logical stream offset (0 = data start) ≠ physical file offset (includes header). Verify `header_offset` parameter in `ParallelReadStreamBuf` constructor.

- Loading throughput collapses after ~50% of a large model:
  Memory pressure: source pages (mmap/page cache) + destination pages (weight buffers) exceed physical RAM. Solution: ensure `ParallelMemStreamBuf` detects and bypasses mmap (Section 3), or use `DONTNEED` on Linux (Section 2).

- Cache miss is slow (first compilation):
  Parallel I/O classes don't help here — they only activate on cache-hit paths. Cache miss slowness is JIT compilation time, not I/O.

- Linux `pread` throughput capped at ~1.5 GB/s despite NVMe being faster:
  Three possible causes: (1) shared fd corrupts readahead state — verify per-thread fd (Section 2), (2) `powersave` CPU governor limiting NVMe controller — check `cpupower frequency-info`, (3) DRAM-less NVMe (HMB) hardware limitation.

- Windows `ReadFile` slower than expected:
  Check if `FILE_FLAG_SEQUENTIAL_SCAN` is used. Verify DLL deployment: stale DLLs in `py310/Lib/site-packages/openvino/libs/` can override newly built versions.

## Implementation Checklist

When adding parallel I/O to a new call site:

1. Determine if it's a file path or memory pointer.
   - File path → `ParallelReadStreamBuf(path, header_offset)`
   - Memory pointer (mmap tensor) → `ParallelMemStreamBuf(data, size)`
2. Verify the link dependency: target must link `openvino::util` (for `ParallelReadStreamBuf`) or the GPU plugin's common_utils.
3. Wrap in `std::istream`: `std::istream stream(&buf);` — existing consumer code stays unchanged.
4. Verify header offset: if the file has a header before data, pass the offset to the constructor.
5. Test with ≥ 4 MB data to trigger parallel path; test with < 4 MB to verify single-read fallback.
6. Measure before and after with cold cache (drop caches: `echo 3 > /proc/sys/vm/drop_caches` on Linux).

## Common Traps

- **Sharing a single fd across `pread` threads on Linux**: Linux kernel maintains per-open-file-description `file_ra_state` for sequential readahead prediction. Concurrent `pread()` on a shared fd corrupts readahead state, collapsing throughput from ~3.5 GB/s to ~0.5 GB/s. Always open a per-thread fd.

- **Using `std::ifstream` for large file reads**: Single-threaded `read()` achieves ~75 MB/s on Windows for 16+ GB files. This is 20× slower than `ParallelReadStreamBuf`. The root cause is single-thread I/O cannot fill NVMe queue depth.

- **Using inline functions in headers that include platform APIs**: `parallel_read_streambuf.hpp` and `parallel_mem_streambuf.hpp` are declaration-only. All implementations live in `.cpp` files. Platform headers (`<unistd.h>`, `<windows.h>`) must NOT leak into headers.

- **Adding `openvino/core/parallel.hpp` dependency to `openvino_util`**: The util library must NOT depend on core. Use `std::thread` directly, not `ov::parallel_for`. This is a deliberate architectural constraint.

- **Assuming mmap is zero-cost**: mmap creates virtual address space mappings, but reading triggers page faults that load pages into both page cache AND process working set. For a 15 GB model, this means 15 GB source + 15 GB destination = 30 GB physical RAM. `ParallelMemStreamBuf`'s file-detection eliminates the source 15 GB entirely.

- **Using pageable `std::vector` as GPU DMA staging buffer**: GPU drivers create implicit pinned-memory bounce buffers, causing hidden CPU copies. Use `allocation_type::usm_host` for zero-copy DMA.

- **Forgetting `posix_fadvise(DONTNEED)` after read on Linux**: Without it, 12+ GB of page cache pages accumulate, causing `kswapd` activation and throughput collapse in the second half of loading. Issue `DONTNEED` per-chunk after `pread` completes.

- **Lowering the 4 MB parallel I/O threshold**: Below 4 MB, thread spawn overhead exceeds I/O time. The threshold is conservative by design.

- **Incorrect header offset → data corruption**: The logical stream sees offset 0 as data start, but the physical file may have a header (e.g., 248 bytes). If `header_offset` is wrong, `pread` reads header bytes as weight data — silent corruption.

- **Windows MSBuild not picking up source changes**: `copy /Y` doesn't always update timestamps. Force recompilation with: `powershell -Command "(Get-Item 'path\to\source.cpp').LastWriteTime = Get-Date"`

## Key Design Invariants

1. **`openvino_util` must not depend on `openvino_core`**: Use `<thread>`, `<vector>`, `<atomic>`, and platform APIs only. No `ov::parallel_for`, no TBB.

2. **Per-thread file descriptors are mandatory** for parallel `pread` on Linux.

3. **4 MB threshold** separates parallel from single-threaded reads. Do not lower without profiling.

4. **`ParallelMemStreamBuf` is a thin adapter**: When file-backed mmap detected → full delegation to `ParallelReadStreamBuf`. All virtual methods forward to `m_file_buf`.

5. **`caching_with_mmap` capability gate** is declared at `plugin.cpp:760`. Changing which plugins advertise this changes which `import_model` overload is called.

6. **Cache miss never reaches either class**: On cache miss, input is `ov::Model&` — no stream, no file read.

7. **Neither class is header-only**: Both have `.cpp` in `src/common/util/src/`, auto-discovered by existing `file(GLOB_RECURSE)`.

## Verified Performance Results

### Windows (Qwen3-30B-A3B, ~15 GB, NVMe)

| Scenario | Baseline | Optimized | Speedup |
|----------|----------|-----------|---------|
| `read_model` 16.3 GB .bin (mmap=false, cold) | 218s (75 MB/s) | 12s (1.3 GB/s) | **18×** |
| Cache blob import (mmap=false) | 39s | 18s | **2.2×** |
| Cache blob import (mmap=true) | 18s | 6s | **2.9×** |
| Pipeline total (mmap=false, cold) | 258s | 33s | **7.9×** |
| Pipeline total (mmap=true) | 19s | 7s | **2.8×** |
| Peak system RAM (mmap=true) | 30.9 GB | 22.3 GB | **−28%** |

### Linux iGPU (12 GB model, 32 GB unified memory)

| Phase | Before | After |
|-------|--------|-------|
| Early (page cache clean) | ~1.5 GB/s | ~2.5-3.5 GB/s |
| Late (page cache pressure) | ~0.5 GB/s | ~1.5-2.5 GB/s |
| Page cache occupancy | 12 GB sustained | ~0 (DONTNEED clears) |

## Source Files

| File | Role |
|------|------|
| `src/common/util/include/openvino/util/parallel_read_streambuf.hpp` | ParallelReadStreamBuf declaration |
| `src/common/util/src/parallel_read_streambuf.cpp` | All ParallelReadStreamBuf implementations |
| `src/plugins/intel_gpu/src/graph/common_utils/parallel_mem_streambuf.hpp` | ParallelMemStreamBuf declaration |
| `src/plugins/intel_gpu/src/graph/common_utils/parallel_mem_streambuf.cpp` | All ParallelMemStreamBuf implementations + platform helpers |
| `src/plugins/intel_gpu/src/plugin/plugin.cpp` | GPU plugin import_model — creates ParallelMemStreamBuf |
| `src/plugins/intel_gpu/src/graph/data.cpp` | GPU weight Host→GPU copy with parallel memcpy |
| `src/inference/src/cache_manager.hpp` | Cache blob read entry (both mmap and non-mmap) |
| `src/inference/src/single_file_storage.cpp` | Single-file cache read with byte offset |
| `src/frontends/ir/src/frontend.cpp` | IR frontend .bin weights read |
| `src/inference/src/dev/core_impl.cpp` | Config plumbing, mmap gate, variant dispatch |
