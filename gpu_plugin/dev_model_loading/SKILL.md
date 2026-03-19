# GPU Model Loading Optimization Skills

This document summarizes the optimization techniques and decisions for improving
model weight/cache loading performance in the OpenVINO GPU plugin.  It
reflects the **final implemented state** on branch `river/mmap_parallel_io_opt`.

---

## 1. Unified Parallel I/O Architecture

### 1.1. Two Code Paths, One Performance Strategy

Cache blob loading in OpenVINO has two entry paths depending on whether
`enable_mmap` is set by the plugin (via `ov::internal::caching_with_mmap`).
Both paths now use a custom `std::streambuf` for parallel I/O:

```
cache.blob ──────────────────────────────────────────── Plugin::import_model()
                                                                   │
 ┌── enable_mmap = true ──────────────────────────────────────────┤
 │  core_impl.cpp:1565                                             │
 │  ov::read_tensor_data(path, blob_offset)                        │
 │  → ov::Tensor (mmap-backed, full blob in process VA space)     │
 │  → Plugin::import_model(const ov::Tensor& model, ...)          │
 │    plugin.cpp:463                                               │
 │    → ParallelMemStreamBuf par_buf(model.data(),                │
 │                                   model.get_byte_size())        │
 │      ┌── if file-backed mmap detected (Linux:/proc/self/maps   │
 │      │                                  Win32:VirtualQuery):    │
 │      │   m_file_buf = make_unique<ParallelReadStreamBuf>(...)  │
 │      │   → all sgetn() calls delegated to ParallelRead         │
 │      └── else (anonymous mmap / USM host buffer):              │
 │          constructor: madvise(MADV_WILLNEED)/PrefetchVM        │
 │          xsgetn(): parallel_for(memcpy, N chunks)              │
 │    → std::istream stream(&par_buf)                              │
 │    → BinaryInputBuffer::read() → rdbuf()->sgetn(dst, n)       │
 │                                                                  │
 └── enable_mmap = false ─────────────────────────────────────────┤
    single_file_storage.cpp:280                                     │
    → ParallelReadStreamBuf par_buf(m_file_path, blob_pos)        │
    → std::istream stream(&par_buf)                                │
    → BinaryInputBuffer::read() → rdbuf()->sgetn(dst, n)         │
      └── ParallelReadStreamBuf::xsgetn()                         │
          < 4 MB: single_read() (one pread loop, no dispatch)     │
          ≥ 4 MB: parallel_read()                                 │
                  N threads × per-thread fd                        │
                  pread(t_fd, dst+offset, chunk, file_off)        │
```

### 1.2. Class Summary

| Class | Header | Impl | Purpose |
|---|---|---|---|
| `ov::util::ParallelReadStreamBuf` | `openvino/util/parallel_read_streambuf.hpp` | `src/common/util/src/parallel_read_streambuf.cpp` | File-backed stream; parallel `pread` (Linux) / `ReadFile` (Windows) from N threads, each with its own fd/HANDLE |
| `ov::util::ParallelMemStreamBuf` | `openvino/util/parallel_mem_streambuf.hpp` | `src/common/util/src/parallel_mem_streambuf.cpp` | Memory-backed stream; detects file-backed mmap and delegates to `ParallelReadStreamBuf`; falls back to `madvise`+`parallel_for(memcpy)` |

Both inherit `std::streambuf`. Reads below 4 MB fall through to single-threaded
calls.  **Neither class is header-only** — both have a corresponding `.cpp` in
`src/common/util/src/`, auto-discovered by the existing `file(GLOB_RECURSE LIBRARY_SRC src/*.cpp)` in the CMakeLists.

### 1.3. Source Files Changed

| File | Change |
|---|---|
| `src/common/util/include/openvino/util/parallel_read_streambuf.hpp` | Declaration-only; no method bodies |
| `src/common/util/src/parallel_read_streambuf.cpp` | All method implementations |
| `src/common/util/include/openvino/util/parallel_mem_streambuf.hpp` | Declaration-only; no method bodies |
| `src/common/util/src/parallel_mem_streambuf.cpp` | All method implementations + platform helpers |
| `src/common/util/CMakeLists.txt` | Added PRIVATE `core/include` path + `ov_set_threading_interface_for` |
| `src/inference/src/single_file_storage.cpp:280` | Replace `std::ifstream`+`seekg` with `ParallelReadStreamBuf(path, blob_pos)` in non-mmap branch |
| `src/plugins/intel_gpu/src/plugin/plugin.cpp:463` | Replace old stream code with `ParallelMemStreamBuf(model.data(), model.get_byte_size())` |
| `src/plugins/intel_gpu/include/intel_gpu/graph/serialization/binary_buffer.hpp:53,117` | Simplified to direct `rdbuf()->sgetn()` — no dynamic-cast detection needed |

---

## 2. CMakeLists.txt Changes

`src/common/util` is a static library (`openvino_util`).  Two additions were
required to compile the new `.cpp` files:

```cmake
# (1) Expose openvino/core/parallel.hpp which lives under core/include
target_include_directories(${TARGET_NAME} PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../core/include>)

# (2) Define OV_THREAD=OV_THREAD_TBB and link TBB privately,
#     eliminating the -Wundef warning from parallel.hpp
ov_set_threading_interface_for(${TARGET_NAME})
```

The PRIVATE qualifier is essential — it avoids creating a reverse dependency
from `openvino_util` on `openvino_core` in the public link interface.

---

## 3. ParallelReadStreamBuf Implementation Details

### 3.1. Why Per-Thread File Descriptors

Each worker thread in `parallel_read` opens its **own** file descriptor via
`open(path, O_RDONLY | O_CLOEXEC)` (Linux) or `CreateFileW(...)` (Windows)
rather than sharing `m_fd` / `m_handle`.

Reason: Linux kernel maintains per-open-file-description `file_ra_state`
(sequential readahead prediction). Multiple concurrent `pread()` calls on a
**shared** fd corrupt each other's readahead state, collapsing throughput from
~3.5 GB/s sequential to ~0.5 GB/s. Each thread having its own fd gives it an
independent readahead window.

### 3.2. Thread Count Selection

```cpp
const size_t hw_threads = parallel_get_max_threads();   // TBB worker count
const size_t max_by_size = size / (1024 * 1024);        // 1 thread per MB
const size_t num_threads = max(1, min(hw_threads, max_by_size));
```

The per-MB heuristic prevents spawning threads for tiny reads (e.g., 3 MB
blob → 3 threads, not 16).

### 3.3. 4 KiB Chunk Alignment

Chunk boundaries are rounded up to 4 KiB so that every thread's file offset
is page-aligned, matching the kernel's page-cache granularity and improving
NVMe I/O coalescing:

```cpp
size_t chunk_size = size / num_threads;
chunk_size = (chunk_size + 4095u) & ~size_t{4095u};
```

Because rounding may create `num_threads × chunk_size > size`, two guards are
applied:
- Non-last threads: `min(chunk_size, size - cur_offset)` — never read past EOF.
- Last thread: `size - cur_offset` — captures every byte including the tail
  fragment that falls beyond `(nthr-1) × chunk_size`.

### 3.4. Thread Dispatch: `parallel_nt_static`

`ov::parallel_nt_static` (not `ov::parallel_for`) is used so:
- Threads are reused from the existing TBB pool (no per-call create/join).
- Lambda receives `(ithr, nthr)` for deterministic partitioning.

An `std::atomic<bool> success{true}` is updated by any failing thread; checked
after the parallel region.

### 3.5. underflow() Buffer

Single-char or line-oriented reads (e.g., `std::getline`, `operator>>`) call
`underflow()`, which batches 8192 bytes via `single_read()`.  This avoids
one `pread()` syscall per character.

### 3.6. Seek Semantics

Internal positions (`m_file_offset`, `m_file_size`) are **absolute file byte
offsets**.  Public positions reported by `seekoff` / `seekpos` are **logical
offsets** (0 = `header_offset`).  The seekoff formula:

```cpp
// ios::cur: account for buffered underflow chars not yet consumed
const streamsize ahead = (gptr() != nullptr) ? (egptr() - gptr()) : 0;
new_pos = m_file_offset - ahead + off;   // stays absolute
return pos_type(m_file_offset - m_header_offset);  // return logical
```

---

## 4. ParallelMemStreamBuf Implementation Details

### 4.1. File-Backed mmap Detection

The constructor immediately checks whether the incoming pointer is backed by a
file on disk.  If yes, it builds a `ParallelReadStreamBuf` over the same
file+offset, which uses direct `pread` instead of mmap page faults.

**Linux** — parse `/proc/self/maps`:
```
start-end perms offset dev inode [pathname]
```
If `addr` falls in a line with a valid filesystem path, `out_offset` is:
`map_file_offset + (addr - range_start)`.

**Windows** — `VirtualQuery(addr, &mbi, ...)`:
- Check `mbi.Type == MEM_MAPPED`.
- `GetMappedFileNameW` → kernel device path like `\Device\HarddiskVolume3\...`.
- `resolve_device_path` (GetLogicalDriveStrings + QueryDosDevice) converts to
  Win32 path `C:\...`.
- `file_offset = (const char*)data - (const char*)mbi.AllocationBase`.

If detection succeeds, all `xsgetn` / `seekoff` / `showmanyc` calls are
**fully delegated** to `m_file_buf` (the `ParallelReadStreamBuf`).

### 4.2. Fallback: parallel_copy

When memory is **not** file-backed (anonymous mmap, USM host buffers):

1. **Prefetch**: `madvise(addr, size, MADV_WILLNEED)` (Linux) /
   `PrefetchVirtualMemory` (Windows) — async I/O while header is being parsed.
2. **Parallel memcpy**: `ov::parallel_for(num_chunks, [&](size_t i) { memcpy... })`.
   - Chunk minimum is 2 MB so small reads stay single-threaded.
   - Windows caps at 16 chunks to bound PFN database lock contention (kernel
     serializes page fault handling via the PFN lock; too many concurrent
     threads cause severe serialization).

### 4.3. Why Not Always Use pread?

`ParallelMemStreamBuf` is created by the GPU plugin from an `ov::Tensor`
(mmap-backed) provided by `core_impl`.  The tensor may point to **anonymous
memory** (e.g., USM host allocations, in-memory blobs) — not just file-backed
mmap.  The file-backed detection logic handles the common cache-hit case; the
fallback handles everything else without changing the caller.

---

## 5. Linux NVMe Root-Cause Analysis (Mar 2026, LNL iGPU machine)

### 5.1. Hardware Facts (Samsung BM9C1 1TB, PCIe 4.0 x4)

| Measurement | Observed |
|---|---|
| O_DIRECT sequential read (dd 4M bs) | **1.2–1.4 GB/s** |
| Buffered cold pread (page cache) | **0.55 GB/s** |
| Warm page-cache read | 8–23 GB/s |
| CPU governor | `powersave` (400 MHz / 5100 MHz max) |
| HMB (DRAM-less NVMe) | `hmb=1` enabled |
| PCIe link | 16 GT/s × 4 (PCIe 4.0 x4) |
| max_sectors_kb | 128 KB |

The NVMe's rated 3.5 GB/s is unachievable under `powersave` + DRAM-less HMB.
The NVMe hardware is the hard ceiling; adding more threads does **not** help.

### 5.2. Why Buffered pread is 55% Slower than O_DIRECT

Buffered `pread` cold path:
1. Kernel allocates 4 KB page-cache pages (CPU cycles per page × 65,536 pages for 256 MB)
2. DMA: NVMe → page-cache (~213 ms at 1.2 GB/s)
3. CPU memcpy: page-cache → user dst (~13 ms at 20 GB/s)
4. Total: ~452 ms ≈ 0.55 GB/s ← page-alloc syscall overhead is the gap

O_DIRECT path: DMA straight to user buffer; no page allocation overhead → ~1.2 GB/s.

### 5.3. Why Thread Parallelism Has Diminishing Returns on This Hardware

From `pread_bench.py` direct measurements:
```
1-thread buffered pread:     452 ms, 0.552 GB/s
8-thread parallel buffered:  425 ms, 0.588 GB/s   ← almost identical
```
The NVMe internal serialization (DRAM-less HMB FTL lookups) is the bottleneck.
Parallel threads raise IO queue depth but the drive cannot service requests faster.

### 5.4. O_DIRECT Was Analyzed but NOT Implemented

O_DIRECT (`open(O_RDONLY | O_DIRECT)`) was considered during analysis and would
give a 2.2× cold-read improvement (0.55 → 1.2 GB/s).  It was **not implemented**
in the final code.  The current `parallel_read_streambuf.cpp` uses plain
`open(O_RDONLY | O_CLOEXEC)` (buffered).

Reason: O_DIRECT requires 512-byte alignment of `(file_offset, size, dst)`.
The blob header is not guaranteed to be so aligned, requiring a dedicate bounce-buffer
path for misaligned reads that adds significant complexity.  The per-thread-fd
approach already saturates the available hardware bandwidth for typical
multi-GB weight reads.

If O_DIRECT is revisited:
- Profile on the target NVMe under `performance` governor first.
- Use `posix_memalign` for destination buffers; enforce 512-byte alignment for
  the blob format weight section.
- Keep the buffered fallback for header/metadata reads that are typically small.

### 5.5. Remaining Hardware Headroom

- Raise CPU governor: `cpupower frequency-set -g performance` — unlocks NVMe
  controller at full frequency.
- Disable APST: `nvme set-feature /dev/nvmeX -f 0x0c -v 0` — prevents
  high-latency power-state transitions between reads.
- Second-run benefit: warm page-cache gives 8–23 GB/s; a background preload
  thread would help repeated model loads on the same machine.

---

## 6. How to Verify the Optimization

### 6.1. Build

```bash
cd ~/river/model_cache/openvino
source build.sh
```

### 6.2. Run LLM Benchmark

```bash
cd ~/river/openvino.genai/
source venv/bin/activate
cd ~/river/openvino.genai/tools/llm_bench
source runme.sh
```

Expected: cold-load parallel-read throughput ≥ 3 GB/s on Linux warm cache,
≥ 0.5 GB/s on cold NVMe (hardware-limited on this LNL machine).

### 6.3. Performance Baselines

| Scenario | Before | After |
|---|---|---|
| mmap, file-backed (GPU, Linux) | mmap page-fault serial | pread parallel via `ParallelReadStreamBuf` delegation |
| non-mmap (single_file_storage) | `std::ifstream` serial | `ParallelReadStreamBuf` parallel pread |
| warm page cache (any) | single-threaded memcpy | parallel memcpy (memory bandwidth limited) |
| cold NVMe (LNL iGPU, powersave) | ~0.5 GB/s | ~0.55–0.6 GB/s (HW-limited) |

---

## 7. Unit Tests

```bash
cd build-x86_64-release/bin/intel64/Release
./ov_inference_unit_tests --gtest_filter=*SingleFileStorageTest*
./ov_inference_unit_tests --gtest_filter=*ParallelReadStreamBufTest*
./ov_inference_unit_tests --gtest_filter=*ParallelMemStreamBufTest*
```

All 33 `*Parallel*` tests pass on the current branch.

---

## 8. Key Design Invariants for Future Work

1. **`openvino_util` must not create a public dependency on `openvino_core`.**
   Use `PRIVATE` includes only.  The threading macro adds only a compile
   definition + private link against TBB.

2. **Never share a single fd across `pread` threads** — Linux per-file-description
   readahead state (`file_ra_state`, `f_ra`) is corrupted by concurrent pread
   on a shared fd.  Always open a per-thread fd.

3. **Threshold (4 MB) is conservative.**  Below threshold, `single_read` is used.
   Do not lower this without profiling: for small models TBB dispatch overhead
   may exceed the read time gained.

4. **`ParallelMemStreamBuf` is a thin adapter.**  When it detects a file-backed
   mmap it becomes a passthrough to `ParallelReadStreamBuf`.  Only the
   constructor path (detection + `m_file_buf` setup) differs; all virtual
   methods then forward to `m_file_buf`.

5. **The `caching_with_mmap` capability gate** is declared at `plugin.cpp:760`.
   Changing which plugins advertise this capability changes which `import_model`
   overload is called and which streambuf class is used.

6. **Cache miss never reaches either class.** On cache miss,
   `compile_model(ov::Model&)` receives an IR graph object — no stream, no
   file read via these classes.  The parallel I/O classes activate **only on
   cache-hit paths**.

