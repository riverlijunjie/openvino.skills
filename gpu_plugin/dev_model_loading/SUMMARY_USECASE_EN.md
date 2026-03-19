# ParallelReadStreamBuf / ParallelMemStreamBuf — Use Case Summary

## Class Overview

| Class | Header | Purpose |
|---|---|---|
| `ParallelReadStreamBuf` | `src/common/util/include/openvino/util/parallel_read_streambuf.hpp` | A `std::streambuf` backed by a file descriptor. Uses `pread` (Linux) / `SetFilePointerEx`+`ReadFile` (Windows) to issue parallel positional reads, bypassing OS sequential-read limitations. |
| `ParallelMemStreamBuf` | `src/common/util/include/openvino/util/parallel_mem_streambuf.hpp` | A `std::streambuf` backed by a memory pointer (typically from `mmap`). Detects whether the pointer originates from a file-backed mapping; if so, internally creates a `ParallelReadStreamBuf` for parallel `pread` access. Falls back to `madvise(MADV_WILLNEED)` + sequential `memcpy` for anonymous mappings. |

---

## Design Goals

Both classes exist to accelerate the **cache-hit path** (2nd and subsequent loads of the same model). They are **never** instantiated during a cache miss (first compilation).

- `ParallelReadStreamBuf` avoids the single-thread sequential I/O bottleneck when reading large `.blob` files from disk.
- `ParallelMemStreamBuf` avoids concurrent page-fault contention that occurs when multiple threads access a file-backed mmap region simultaneously; it delegates to `ParallelReadStreamBuf` to use `pread` directly, which bypasses the page-fault path entirely.

A size threshold (`DEFAULT_THRESHOLD = 4 MB`) controls whether parallel or single-threaded reads are used, preventing scheduling overhead from exceeding I/O gains for small blobs.

---

## Use Case Matrix

| Scenario | `enable_mmap` | Plugin supports `caching_with_mmap` | Class(es) Used | Source Location |
|---|---|---|---|---|
| **Cache hit — separate-file cache, non-mmap** | `false` | N/A | `ParallelReadStreamBuf` | `cache_manager.hpp:71` |
| **Cache hit — separate-file cache, mmap** | `true` | ✅ (e.g., GPU) | `ov::Tensor` mmap → plugin creates `ParallelMemStreamBuf` → internally `ParallelReadStreamBuf` | `cache_manager.hpp:62` + `plugin.cpp:460` |
| **Cache hit — single-file cache, non-mmap** | `false` | N/A | `ParallelReadStreamBuf` (with byte offset) | `single_file_storage.cpp:280` |
| **Cache hit — single-file cache, mmap** | `true` | ✅ (e.g., GPU) | `ov::Tensor` mmap → plugin creates `ParallelMemStreamBuf` → internally `ParallelReadStreamBuf` | `single_file_storage.cpp:263` + `plugin.cpp:460` |
| **Cache miss (first compilation)** | any | any | **Neither class** — input is `ov::Model&` (IR graph object); only write (`export_model`) happens | — |
| `import_model(istream&, ...)` direct API call | any | any | Determined by the caller; the cache manager is bypassed entirely | — |

---

## Full Call Chains

### Cache Hit — `enable_mmap=false`

```
ov::Core::compile_model(model_path, "GPU", config)
  └─ CoreImpl::compile_model_and_cache_impl()
       └─ CacheManager::read_cache_entry(blob_path)        // cache_manager.hpp:71
            └─ ParallelReadStreamBuf par_buf(blob_path)    // multi-threaded pread
                 └─ std::istream stream(&par_buf)
                      └─ plugin.import_model(stream, config)
                           └─ BinaryInputBuffer(stream, engine)
                                └─ deserialize GPU kernels
```

### Cache Hit — `enable_mmap=true` (GPU, advertises `caching_with_mmap`)

```
ov::Core::compile_model(model_path, "GPU", config)
  └─ CoreImpl::compile_model_and_cache_impl()
       └─ [gate] m_mmap_enabled && device_supports_internal_property(caching_with_mmap)
                                                            // core_impl.cpp:1565
            └─ CacheManager::read_cache_entry(blob_path)   // cache_manager.hpp:62
                 └─ ov::read_tensor_data(blob_path) → ov::Tensor  // OS mmap entire file
                      └─ std::visit(model_importer, CompiledBlobVariant)
                                                            // core_impl.cpp:1566
                           └─ plugin.import_model(Tensor, config)
                                                            // plugin.cpp:460
                                └─ ParallelMemStreamBuf mem_buf(tensor.data(), size)
                                     ├─ [Linux] parse /proc/self/maps for file path
                                     │    └─ file-backed → new ParallelReadStreamBuf(path)
                                     │         └─ parallel pread (bypasses page-fault)
                                     └─ [fallback] madvise(MADV_WILLNEED) + memcpy
                                          └─ BinaryInputBuffer(stream, engine)
                                               └─ deserialize GPU kernels
```

### Cache Miss — Any Configuration

```
ov::Core::compile_model(model_path, "GPU", config)
  └─ CoreImpl::compile_model_and_cache_impl()
       └─ plugin.compile_model(ov::Model&, config)   // input is parsed IR graph, NOT a stream
            └─ GPU: JIT-compile all OpenCL kernels
                 └─ export_model(std::ofstream)       // write-only: save blob to cache
                      // ParallelReadStreamBuf and ParallelMemStreamBuf are never instantiated
```

---

## Key Gating Conditions

| Condition | Code Location |
|---|---|
| `enable_mmap` user config propagation | `CoreImpl` → `CoreConfig` → `CacheContent.m_mmap_enabled` |
| mmap path activation gate | `core_impl.cpp:1565`: `m_mmap_enabled && device_supports_internal_property(plugin, ov::internal::caching_with_mmap)` |
| GPU plugin advertises mmap support | `plugin.cpp:760`: returns `caching_with_mmap` capability |
| `CompiledBlobVariant` dispatch | `core_impl.cpp:1566`: `std::visit(model_importer, compiled_blob)` — index 0 = `ov::Tensor` (mmap path), index 1 = `std::istream&` (stream path) |
| Linux file-backed mmap detection | `parallel_mem_streambuf.cpp`, anonymous namespace `get_mmap_file_info()`: parses `/proc/self/maps` |
| Windows file-backed mmap detection | `parallel_mem_streambuf.cpp`, anonymous namespace `resolve_device_path()`: `VirtualQuery` + `GetMappedFileNameW` |
| Parallel vs. single-thread threshold | `ParallelReadStreamBuf::DEFAULT_THRESHOLD = 4 * 1024 * 1024` bytes |

---

## Interaction Between the Two Classes

`ParallelMemStreamBuf` **optionally owns** a `ParallelReadStreamBuf` as an internal member (`m_file_buf`):

```
ParallelMemStreamBuf (wraps void* + size)
    │
    ├─ constructor: detect if pointer is file-backed mmap
    │       ↓ yes
    │   m_file_buf = std::make_unique<ParallelReadStreamBuf>(detected_path,
    │                                                         header_offset)
    │       ↓ no
    │   madvise(MADV_WILLNEED) / PrefetchVirtualMemory (hint OS)
    │
    └─ xsgetn / underflow / seekoff
            ├─ if m_file_buf → delegate all reads to ParallelReadStreamBuf
            └─ else → parallel_copy from memory (ov::parallel_for + memcpy)
```

This design means the two classes are not peers — `ParallelMemStreamBuf` is the consumer-facing entry point for the mmap code path, and it **reuses** `ParallelReadStreamBuf` internally when the mmap pointer is traceable to a file on disk.

---

## Performance Rationale

| Scenario | Without Optimization | With Optimization |
|---|---|---|
| Large `.blob` from disk (`enable_mmap=false`) | Single-threaded sequential `ifstream::read` | Multi-threaded `pread` splits the file into N chunks, one per TBB thread |
| File-backed mmap (`enable_mmap=true`) | Concurrent `memcpy` from mmap region triggers parallel OS page faults; kernel lock contention | Parallel `pread` skips the mmap page-fault entirely; data read directly into destination buffer |
| Anonymous mmap / small blob | N/A | `madvise(MADV_WILLNEED)` issued as prefetch hint; `memcpy` proceeds normally (no `pread`) |
| Cache miss | N/A | Neither class involved; JIT compilation dominates |

---

## Files Modified in `river/mmap_parallel_io_opt` Branch

| File | Change |
|---|---|
| `src/common/util/include/openvino/util/parallel_read_streambuf.hpp` | Declaration only (no method bodies) |
| `src/common/util/src/parallel_read_streambuf.cpp` | All method implementations |
| `src/common/util/include/openvino/util/parallel_mem_streambuf.hpp` | Declaration only (no method bodies, no platform headers) |
| `src/common/util/src/parallel_mem_streambuf.cpp` | All method implementations + platform helper functions in anonymous namespace |
| `src/common/util/CMakeLists.txt` | Added PRIVATE core include path for `openvino/core/parallel.hpp`; added `ov_set_threading_interface_for(${TARGET_NAME})` to fix `-Wundef` on `OV_THREAD` |
