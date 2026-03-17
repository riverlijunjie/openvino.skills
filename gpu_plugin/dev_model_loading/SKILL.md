# GPU Model Loading Optimization Skills

This document summarizes the optimization techniques and considerations for improving model weight/cache loading performance in the OpenVINO GPU plugin.

## 0. API Overview



## 1. IO Bottleneck Analysis
*   **Problem**: Loading large model weights (GBs) via standard C++ `std::istream` is single-threaded and CPU-bound due to memory copies and kernel buffering.
*   **Symptom**: Throughput caps at ~1GB/s on NVMe drives capable of 3.5GB/s+.
*   **Goal**: Saturate disk IO bandwidth and reduce model startup time.

## 2. Linux Optimization: Zero-Copy (Direct IO)
*   **Technique**: Use `O_DIRECT` to bypass the OS PageCache and copy data directly from disk to user-space memory.
*   **Implementation Details**:
    *   **File Descriptor Extraction**: Hack `std::filebuf` to extract the underlying file descriptor (fd) from the stream using `_M_file.fd()` (GCC specific) or `/proc/self/fd/`.
    *   **Aligned Memory**: Ensure the destination buffer is aligned to the block size (usually 4096 bytes) using `posix_memalign`.
    *   **Execution**: Use `pread` for thread-safe positional reads without modifying the file pointer.
*   **Fallback**: If buffer alignment or file system constraints fail `O_DIRECT`, fall back to standard buffered IO.

## 3. Windows Optimization: Native Parallel IO
*   **Technique**: Multi-threaded reading using Native Win32 APIs (`CreateFileW`, `ReadFile`) to utilize NVMe multiple queues.
*   **Why not std::thread + ifstream?**: `std::ifstream` has internal locks and shared state that severely limit parallel scaling.
*   **Implementation Details**:
    *   **Stateless Reading**: Use `OVERLAPPED` structure to specify offsets for each read, allowing lock-free concurrent execution.
    *   **File Sharing**: Open files with `FILE_SHARE_READ | FILE_SHARE_WRITE` to allow concurrent access by other processes (e.g., cache writers).
    *   **Unicode Support**: Always convert `std::string` paths to `std::wstring` (WideChar) for `CreateFileW`.
    *   **Chunking**: Split the read request into 4KB-aligned chunks distributed across threads (e.g., `std::thread::hardware_concurrency()`).

## 4. Critical Logic: Offset Compensation
*   **Problem**: "Garbled Data" / Corruption.
    *   The logical `std::istream` (OpenVINO stream) often views a sub-section of the file (data only), effectively having an offset of 0.
    *   The physical file on disk may contain a Header (e.g., 248 bytes) before the data.
    *   Naive `load_parallel` opens the physical file at physical offset 0, reading headers as data.
*   **Solution**: Automatic Header Detection.
    1.  Get **Logical Stream Size** (`seekg(0, end)`).
    2.  Get **Physical File Size** (`GetFileSizeEx` / `stat`).
    3.  Calculate `Delta = Physical - Logical`.
    4.  Add `Delta` to all read offsets passed to the low-level loader.

## 5. General Best Practices
*   **Thresholding**: Only trigger complex IO optimizations for large data blocks (e.g., >4MB). Small metadata reads perform better with standard buffered IO.
*   **Memory alignment**: Always align buffers (4096 bytes) to satisfy Direct IO requirements on both Linux and Windows.
*   **Atomic Error Handling**: Use `std::atomic<bool>` for status flags in multi-threaded loaders to ensure fast failure propagation.
## 6. Discrete GPU Loading Techniques (`is_alloc_host_accessible = false`)
When copying loaded model weights onto a discrete GPU, further architectural optimizations must be applied to prevent host-side bottlenecks:

*   **USM (Unified Shared Memory) Staging Buffers**: Never use pageable memory (e.g., `std::vector`) as intermediate buffers for GPU DMA. GPU drivers implicitly create a pinned memory bounce buffer, causing an expensive hidden CPU copy. Instead, directly allocate a Host USM buffer (`allocation_type::usm_host`). The CPU can read into it, and the GPU can DMA fetch it directly without intermediary copies (Zero-Copy).
*   **Double-Buffering (Ping-Pong Pipeline)**: To perfectly overlap CPU disk reading and GPU PCIe transfer, maintain two identical USM buffers. While the CPU acts on local fast I/O into Buffer A, the GPU performs an asynchronous DMA DMA from Buffer B to VRAM.
*   **L3 Cache-Friendly Pipeline Chunk Sizing**: Rather than buffering massive chunks (e.g., 32MB+), tune the `DATA_BLOCK_SIZE` to fit inside the CPU's Last Level Cache (e.g., **4MB**). This guarantees that right after the CPU completes the disk I/O, the memory block is still hot in the CPU's L3 cache. The subsequent GPU DMA will snoop the PCIe cache coherency and pull data natively from L3 instead of slow system DDR.
*   **Zero-Wait Asynchronous Launch**: Whenever possible, avoid host-side `wait()` commands that stall the CPU during DMA execution. Pass the DMA events down the OpenVINO graph to natively schedule neural network kernels on the same asynchronous command queue once the memory transfer finishes.

## 7. Memory Mapped (mmap) Tensor Parallel Reading
*   **Problem**: In earlier implementations, custom file paths or low-level streams were manually created to bypass `std::istream` bottlenecks (which force serialization and internal buffering). This approach disabled or clashed with the built-in `mmap` functionality of `ov::Tensor`.
*   **Requirement**: Stop bypassing `mmap` with ad-hoc physical file queries. When `mmap` is enabled, the OS kernel already maps the file to a contiguous virtual memory space, wrapped as an `ov::Tensor` inside the `Plugin::import_model` flow.
*   **Solution**:
    1.  **Extract Pointer**: Identify if the incoming `std::istream` is backed by an `ov::SharedStreamBuffer` (indicative of a mapped tensor). If so, extract the underlying raw memory pointer (`get_ptr()`).
    2.  **Multithreaded Memcpy**: Once the raw pointer is obtained, chunk the buffer (e.g., 2MB segments) and use `ov::parallel_for` to invoke `std::memcpy` concurrently.
    3.  **NVMe Saturation**: Multiple threads triggering concurrent page faults (Page Cache misses) against the mapped memory file forces the OS kernel to dispatch concurrent I/O requests down to the NVMe driver. This elevates the hard disk Queue Depth and saturates NVMe PCIe bandwidth, completely avoiding `std::istream` locks and single-threaded stall.
*   **Implementation Steps**:
    *   Add `get_ptr()` and `size()` accessors in `src/core/dev_api/openvino/runtime/shared_buffer.hpp`.
    *   Intercept calls in `BinaryInputBuffer::read()` to dynamically cast `_stream.rdbuf()` -> `ov::SharedStreamBuffer*`.
    *   Execute an `ov::parallel_for` memory copy mechanism and update the stream offset correctly.

### 7.1. Eliminating Pageable Bounce-Buffers via USM (is_alloc_host_accessible=false)
For the discrete GPU path where memory is inaccessible to the host naturally, OpenVINO utilizes double-buffering DMA data transfer inside primitives like `data.hpp` (`load_weights`). 
*   **A major anti-pattern**: Using standard `std::vector<uint8_t>` for intermediate double buffers forces the GPU drivers to create internal locked bounce-buffers, drastically increasing CPU overhead during PCIe DMA transfers.
*   **The Correct Fix**: Completely replace `std::vector` inside the fallback path logic with zero-copy Unified Shared Memory (`allocation_type::usm_host`). By allocating host memory using the target engine, we directly create page-locked pinned memory that is immediately legible for the driver, satisfying real zero-copy transfers across the pipeline overhead.


## 8. Universal Cross-Plugin Optimization: Parallel-IO `std::streambuf` Wrapper

### 8.1. Motivation

The optimizations in sections 2, 3, and 7 all share a fundamental limitation: they are applied
**at the consumption site** (inside `BinaryInputBuffer::read()` or `data.hpp::load_weights()`).
This means every plugin that reads a serialized cache blob (`intel_gpu`, `intel_cpu`, `intel_npu`,
etc.) must independently discover and implement the same tricks.

When `mmap` is disabled (or not available), the blob file is opened as a plain `std::ifstream`
and passed down as a `std::istream&`. The bottleneck reverts to single-threaded `sgetn()` regardless
of any per-plugin effort, because the stream layer itself is the serialization point.

### 8.2. Design: `ParallelReadStreamBuf`

Create a custom `std::streambuf` subclass that internally uses parallel I/O for large reads,
exposing a standard `std::istream`-compatible interface.  Any code that currently accepts a
`std::istream&` gets the speedup transparently — no per-plugin changes required.

```
┌──────────────────────────────────────────────────┐
│  BinaryInputBuffer::read()  (unchanged fallback)  │
│    _stream.rdbuf()->sgetn(dst, size)              │
└───────────────────┬──────────────────────────────┘
                    │ virtual sgetn() dispatch
                    ▼
┌──────────────────────────────────────────────────┐
│  ParallelReadStreamBuf  : public std::streambuf   │
│                                                   │
│  Fields:                                          │
│    int fd_ / HANDLE hFile_  (native handle)       │
│    size_t file_offset_       (current position)   │
│    size_t threshold_         (e.g. 4 MB)          │
│                                                   │
│  Overrides:                                       │
│    xsgetn(char* dst, streamsize n)                │
│      if n >= threshold_:                          │
│        → parallel_read(fd_, dst, n, file_offset_) │
│          (pread × N threads on Linux,             │
│           ReadFile(OVERLAPPED) × N on Windows)    │
│      else:                                        │
│        → standard read() / ReadFile()             │
│    seekoff() / seekpos()  — update file_offset_   │
│    showmanyc()            — return remaining size │
└──────────────────────────────────────────────────┘
```

### 8.3. Key Design Points

*   **`xsgetn` is the right override.** `std::streambuf::xsgetn` is the virtual method called
    by `sgetn()`. Overriding it bypasses the internal get-area buffer entirely, allowing direct
    large DMA-style reads into the caller's `dst` without any intermediate copy.
*   **No internal buffer for large reads.** For `n >= threshold`, write directly into `dst`
    using parallel low-level reads. For `n < threshold`, fall back to a small internal buffer
    (or a single `read()`/`ReadFile()` call) to keep metadata reads efficient.
*   **File handle ownership.** The wrapper opens its own native handle (fd on Linux,
    `HANDLE` on Windows) from the same file path, independent of any `std::ifstream` already
    in use. This avoids sharing the seek pointer and enables concurrent reads without locks.
*   **Offset tracking.** Maintain `file_offset_` manually. `xsgetn` advances it by `n` after
    each read. `seekoff`/`seekpos` update it directly. This replaces `lseek`/`SetFilePointer`
    calls inside each read thread (use absolute-offset APIs — `pread`, `OVERLAPPED` — instead).
*   **`showmanyc` override.** Return `file_size_ - file_offset_` so that `std::istream::read`
    can avoid unnecessary internal `underflow()` calls for large sequential reads.
*   **Header offset compensation.** If the blob file has a fixed header before the data region,
    apply the offset delta at construction time (same logic as Section 4) so callers see a
    logical offset-0 view of the data.

### 8.4. Integration Point

```cpp
// In Plugin::import_model() or wherever the cache stream is created:
auto parallel_buf = std::make_shared<ParallelReadStreamBuf>(cache_file_path, header_offset);
std::istream parallel_stream(parallel_buf.get());
// Pass parallel_stream wherever std::istream& is expected — BinaryInputBuffer, etc.
```

This is the **lowest-level injection point** possible: every downstream consumer (`BinaryInputBuffer`,
all plugin deserialization paths, `EncryptedBinaryInputBuffer` wrapping) automatically gets
parallel I/O without any change.

### 8.5. Trade-offs vs. Per-Site Approaches

| Dimension | Per-site (current) | `ParallelReadStreamBuf` |
|---|---|---|
| Coverage | GPU plugin only | All plugins, any consumer of `std::istream` |
| mmap compatibility | Requires mmap or `weights_path` | Works with plain file streams |
| Encrypted cache | Not applicable | Compatible (encrypt layer wraps the fast buffer) |
| Code change scope | `binary_buffer.hpp`, `data.hpp` | New class in `src/common/util/` |
| Complexity | Low per file, high total | Concentrated in one class |
| Requires `weights_path` | Yes (Approach B) | No — only the file path at open time |
| Platform | Windows-specific (`ReadFile`) | Cross-platform (`pread` / `ReadFile`) |

### 8.6. Risks and Mitigations

*   **`seekable` requirement**: The implementation assumes the underlying file is seekable (NVMe/SSD).
    For non-seekable streams (pipes, sockets), detect at construction (`lseek` returns -1) and
    fall back to standard single-read behavior.
*   **Small-read regression**: Parallel overhead is significant for reads smaller than ~1 MB.
    Always gate on `threshold_` (default 4 MB) and fall back to a buffered single-read for
    metadata fields.
*   **Thread pool management**: Spawning N threads per `xsgetn` call adds latency for the first
    block. Use a pre-created thread pool (or `ov::parallel_for` via TBB) to amortize creation
    cost across blocks.

---

## 9. Unified Parallel I/O Architecture (Current Implementation)

### 9.1. Two Paths, One Strategy

Cache blob loading has two entry points depending on whether `enable_mmap` is set
by the plugin (via `ov::internal::caching_with_mmap`):

```
cache.blob ─────────────────────────────────────────────────────────────── Plugin::import_model()
                                                                                      │
    ┌─── enable_mmap = true ──────────────────────────────────────────────────────────┤
    │   ov::read_tensor_data(path, blob_offset)                                       │
    │   → ov::Tensor (mmap-backed)                                                    │
    │   → Plugin::import_model(const Tensor&, ...)                                   │
    │     → ParallelMemStreamBuf(tensor.data(), tensor.size())  ◄── NEW              │
    │     → std::istream stream(&par_buf)                                             │
    │     → import_model(stream, ...)  → BinaryInputBuffer → sgetn() ──────────── weights
    │       └── ParallelMemStreamBuf::xsgetn() for large reads                       │
    │           PrefetchVirtualMemory + ov::parallel_for(memcpy) + VirtualFree(MEM_RESET)
    │                                                                                 │
    └─── enable_mmap = false ────────────────────────────────────────────────────────┤
        single_file_storage.cpp                                                       │
        → ParallelReadStreamBuf(path, blob_offset)  ◄── NEW                         │
        → std::istream stream(&par_buf)                                               │
        → import_model(stream, ...)  → BinaryInputBuffer → sgetn() ──────────── weights
          └── ParallelReadStreamBuf::xsgetn() for large reads                        │
              pread(fd, dst, n, offset) × N threads (Linux)                          │
              ReadFile(OVERLAPPED) × N threads (Windows)                             │
```

### 9.2. Classes Implemented

| Class | Header | Purpose |
|---|---|---|
| `ov::util::ParallelReadStreamBuf` | `openvino/util/parallel_read_streambuf.hpp` | File-backed stream; direct `pread`/`ReadFile` from N threads |
| `ov::util::ParallelMemStreamBuf` | `openvino/util/parallel_mem_streambuf.hpp` | Memory-backed stream; parallel `memcpy` from mmap tensor data |

Both inherit `std::streambuf` and override `xsgetn()`. Reads below 4 MB fall through
to a single-threaded call. Both classes are header-only in `src/common/util/`.

### 9.3. Integration Points Changed

| File | Change |
|---|---|
| `src/inference/src/single_file_storage.cpp` | Replace `std::ifstream`+`seekg` with `ParallelReadStreamBuf(path, blob_pos)` in non-mmap branch |
| `src/plugins/intel_gpu/src/plugin/plugin.cpp` | Replace `SharedStreamBuffer` with `ParallelMemStreamBuf` in `import_model(Tensor, ...)` |
| `src/plugins/intel_gpu/include/intel_gpu/graph/serialization/binary_buffer.hpp` | Remove `SharedStreamBuffer` dynamic-cast detection; simplified to single `sgetn()` call |

### 9.4. mmap Path: Why Not Skip mmap Entirely?

On Windows, `ParallelReadStreamBuf` (direct `ReadFile`) achieves ~4.15 GB/s vs
`ParallelMemStreamBuf` (mmap + parallel memcpy) at ~1.89 GB/s.  The mmap path
is still used when the plugin explicitly declares `caching_with_mmap` support,
because:
- Plugins may rely on lazy-loading semantics of mmap (pages not faulted until accessed).
- Changing from tensor variant to stream variant alters the `import_model` overload
  called, which may affect other plugins differently.
- The `VirtualFree(MEM_RESET)` mitigation in `ParallelMemStreamBuf` recovers most
  of the working-set pressure that caused the observed Windows stalls.

If further mmap-path improvement is needed, `single_file_storage.cpp` can
conditionally replace the mmap tensor with a `ParallelReadStreamBuf` stream for
plugins where the two `import_model` overloads are equivalent (GPU plugin).

