# Model Loading I/O Optimization: Comparative Analysis Across ML Frameworks

## Executive Summary

This report analyzes how major ML framework ecosystems optimize SSD/NVMe read bandwidth
during model loading. The key finding is that mature frameworks converge on a small set
of strategies: **mmap-based zero-copy**, **O_DIRECT bypass of page cache**, 
**async GPU upload pipelines**, and **format-level design for contiguous tensor access**.

OpenVINO's `ParallelReadStreamBuf` approach (multi-thread pread with per-thread file
descriptors) is unique among these frameworks and addresses a specific limitation of
mmap — namely, the overhead of page faults and PTE creation into process RSS. No other
framework in this analysis implements parallel `pread()` for model loading.

---

## 1. llama.cpp — Most Sophisticated I/O Pipeline

**Source**: `llama-model-loader.cpp` (~1686 lines), `llama-mmap.h`

### 1.1 Three Loading Modes

llama.cpp provides three distinct model loading strategies, selectable at runtime:

| Mode | Flag | Mechanism | Page Cache Impact |
|------|------|-----------|-------------------|
| **mmap** | `use_mmap=true` | `mmap()` + optional `mlock()` | Uses page cache, data mapped into PTE |
| **Direct I/O** | `use_direct_io=true` | `O_DIRECT` / `FILE_FLAG_NO_BUFFERING` | Bypasses page cache entirely |
| **Standard read** | Both false | `fread()` / `ReadFile()` | Populates page cache, copies to user buffer |

**Priority rule**: Direct I/O takes precedence over mmap. When `use_direct_io` is
available and requested, mmap is automatically disabled:
```cpp
// from llama-model-loader constructor
if (use_direct_io) {
    use_mmap = false;  // direct I/O wins
}
```

### 1.2 mmap with NUMA Awareness

For CPU inference, llama.cpp uses mmap as the default for zero-copy tensor access:
- Tensor data pointers reference the mmap'd region directly (no copy for CPU tensors)
- `llama_mmap` wraps OS-specific mmap: `mmap()` on POSIX, `CreateFileMapping()`+`MapViewOfFile()` on Windows
- `llama_mlock` pins pages to avoid swap-out during inference
- Unused regions are unmapped after loading via `unmap_fragment()` to reduce RSS
- NUMA-aware allocation: mmap'd data can be bound to specific NUMA nodes

### 1.3 Async GPU Upload Pipeline (Key Innovation)

For GPU inference, llama.cpp implements an **asynchronous multi-buffer upload pipeline**
that maximizes NVMe → GPU bandwidth:

```
Architecture:
  [NVMe SSD] --read()--> [Staging Buffer 0] --cudaMemcpyAsync--> [GPU]
  [NVMe SSD] --read()--> [Staging Buffer 1] --cudaMemcpyAsync--> [GPU]
  [NVMe SSD] --read()--> [Staging Buffer 2] --cudaMemcpyAsync--> [GPU]
  [NVMe SSD] --read()--> [Staging Buffer 3] --cudaMemcpyAsync--> [GPU]
```

- **4 pinned staging buffers** × **64 MB each** for NVMe drives (1 MB for non-direct I/O)
- Buffer size: `alignment != 1 ? 64*1024*1024 + 2*alignment : 1*1024*1024`
- Double-buffering pattern: while GPU is consuming buffer N, CPU reads into buffer N+1
- Uses `cudaMemcpyAsync()` for non-blocking GPU transfers
- Alignment-aware reads: `read_aligned_chunk()` ensures O_DIRECT alignment requirements

### 1.4 File Abstraction (`llama_file`)

The `llama_file` class provides a unified I/O interface:
- `read_alignment()`: returns the required alignment for direct I/O
- `has_direct_io()`: checks if O_DIRECT is available and enabled
- `read_raw_unsafe()`: low-level unbuffered read
- `read_aligned_chunk()`: alignment-padded read for O_DIRECT compliance

### 1.5 Split Model Support

Large models can be split across multiple GGUF files with coordinated loading,
allowing parallel I/O from multiple physical drives.

---

## 2. safetensors (Hugging Face) — Format-Level Optimization

**Source**: `tensor.rs` (~1553 lines), README.md

### 2.1 Zero-Copy File Format

safetensors achieves I/O efficiency primarily through **format design** rather than
runtime I/O tricks:

```
File layout:
  [8 bytes: header_size] [JSON header] [contiguous tensor data...]
```

- **8-byte header prefix**: enables instant header size detection
- **JSON metadata header**: tensor names, shapes, dtypes, and byte offsets
- **Contiguous packed tensors**: all tensor data laid out sequentially after header
- No pickle, no arbitrary code execution (security benefit)

### 2.2 mmap-Based Deserialization

```rust
let file = File::open(filename)?;
let buffer = unsafe { MmapOptions::new().map(&file)? };
let tensors = SafeTensors::deserialize(&buffer)?;
```

- Uses `memmap2::MmapOptions` for memory-mapped file access
- Deserialization returns metadata + data slice references (no copy)
- Individual tensors can be loaded without scanning the entire file
- ~400 lines of code vs ~210,000 lines for HDF5

### 2.3 macOS Write Optimization

```rust
#[cfg(target_os = "macos")]
unsafe { libc::fcntl(file.as_raw_fd(), libc::F_NOCACHE, 1) };
```

`F_NOCACHE` bypasses macOS page cache for write operations, achieving ~30% write
improvement according to the README.

### 2.4 Limitations

- **No parallel read mechanism**: relies entirely on mmap + OS page cache behavior
- **No direct I/O support**: page cache bypass not implemented for reads
- **No async GPU upload**: no GPU-specific optimizations
- Optimization comes from format simplicity, not I/O sophistication

---

## 3. PyTorch — mmap + Meta Device Pattern

**Source**: `torch.load` documentation, PyTorch loading tips tutorial

### 3.1 `torch.load(mmap=True)` (Since 2.1.0)

PyTorch added mmap support for lazy tensor storage loading:

```python
state_dict = torch.load("model.pt", mmap=True, weights_only=True)
```

- Lazily loads tensor storages via mmap instead of reading entire file
- Metadata loading: ~0.004s with mmap vs ~0.035s without (9× faster for metadata)
- Internally uses `torch.UntypedStorage.from_file` for zero-copy CPU access

### 3.2 Meta Device + Assign Pattern

The recommended pattern for minimal memory model loading:

```python
# 1. Create model skeleton without allocating memory
with torch.device('meta'):
    model = MyModel()

# 2. Load weights via mmap (lazy, no immediate physical read)
state_dict = torch.load("model.pt", mmap=True, weights_only=True)

# 3. Assign by reference instead of in-place copy (avoids 2× memory)
model.load_state_dict(state_dict, assign=True)
```

Key insight: `assign=True` replaces parameter references rather than copying data
into pre-allocated buffers, halving peak memory usage.

### 3.3 Limitations

- **No parallel I/O**: sequential pickle/zip-based deserialization
- **No direct I/O**: entirely page-cache dependent
- **No async GPU upload**: data moves to GPU during `model.to(device)` synchronously
- **Pickle format**: inherent overhead from Python serialization format
- Optimization is about reducing memory copies, not increasing I/O bandwidth

---

## 4. vLLM — Tensorizer + safetensors Integration

**Source**: `tensorizer.py` (~794 lines), `utils.py` (~287 lines)

### 4.1 CoreWeave Tensorizer (Primary Optimization)

vLLM integrates CoreWeave's `tensorizer` library for high-performance model
deserialization:

```python
from tensorizer import TensorDeserializer
from tensorizer.stream_io import open_stream

with open_stream(uri, mode="rb", **stream_kwargs) as stream:
    with TensorDeserializer(stream, dtype=dtype,
                            device=f"cuda:{device_index}",
                            **deserialization_kwargs) as deserializer:
        deserializer.load_into_module(model)
```

Key characteristics:
- **Direct GPU loading**: `device=f"cuda:{device_index}"` enables direct-to-GPU
  deserialization without CPU staging
- **Stream-based I/O**: supports local files, S3, HTTP/HTTPS URIs
- **Encryption support**: `DecryptionParams`/`EncryptionParams` for secure model storage
- Custom serialization format optimized for streaming deserialization

### 4.2 Preferred Loading Path

vLLM uses a "vLLM-tensorized" model format that serializes the model after vLLM's
internal tensor parallelism setup:
- Pre-sharded weights: each GPU rank loads only its shard
- `no_init_or_tensor` context: avoids allocating memory before loading
- `MetaTensorMode()`: creates parameter shells on meta device

### 4.3 safetensors Fallback

For non-tensorized models, vLLM falls back to safetensors format loading:
- Generates weight iterators via `tensorizer_weights_iterator()`
- Warns that HuggingFace model loading is "not optimized for vLLM"
- Forces CPU loading, then moves to GPU (suboptimal)

### 4.4 Device Loading Context

```python
with device_loading_context(module, target_device):
    # load and process weights with quantization
    process_weights_after_loading(model)
```

Manages CPU↔GPU parameter movement during quantization-aware weight processing.
UVA (Unified Virtual Addressing) offloading with `pin_memory` support for
efficient GPU memory management.

---

## 5. ONNX Runtime — Protobuf/Flatbuffer Based

**Source**: `inference_session.cc` (~4115 lines), `onnxruntime_session_options_config_keys.h`

### 5.1 Model Loading Architecture

ONNX Runtime supports two model formats with different I/O characteristics:

| Format | Loading Method | I/O Optimization |
|--------|---------------|------------------|
| **ONNX (.onnx)** | Protobuf `ParseFromArray()` | Full file read + protobuf deserialization |
| **ORT (.ort)** | FlatBuffers zero-copy | Can use model bytes directly (zero-copy) |

### 5.2 ORT Format Optimizations

The ORT format (FlatBuffers-based) provides several I/O-related options:

- **`use_ort_model_bytes_directly`**: Disables copying model bytes during session
  creation. The caller must guarantee the buffer remains valid:
  ```cpp
  ort_format_model_bytes_ = gsl::span<const uint8_t>(
      reinterpret_cast<const uint8_t*>(model_data), model_data_len);
  ```

- **`use_ort_model_bytes_for_initializers`**: Uses FlatBuffer bytes directly for
  initializer tensors, avoiding allocation + copy. Requires
  `use_ort_model_bytes_directly` to be enabled.

### 5.3 External Data and Initializers

For models with large weights stored externally:
- **External initializers file path** (`model_external_initializers_file_folder_path`):
  Supports loading initializers from separate data files
- **External data loader manager** (`ExternalDataLoaderManager`): EPs can register
  custom data loaders for external initializer data
- **Pre-packed weights**: Can be saved to external files and memory-mapped on next load,
  trading disk space for faster initialization

### 5.4 Standard Loading Path (No Parallel I/O)

The default ORT file loading in `LoadOrtModelBytes()` is straightforward:
```cpp
std::ifstream bytes_stream(model_uri, std::ifstream::in | std::ifstream::binary);
bytes_stream.read(reinterpret_cast<char*>(bytes_data_holder.data()), num_bytes);
```

- Single-threaded sequential `ifstream::read()`
- No mmap, no direct I/O, no parallel reading
- Page cache dependent through standard POSIX I/O stack

### 5.5 Limitations

- **No mmap support for standard model loading**: relies on full file read
- **No parallel I/O**: single-thread per-session model loading
- **No direct I/O**: all reads go through the page cache
- ORT format + direct bytes usage is the closest to zero-copy, but requires specific
  session configuration

---

## 6. TensorRT — Engine Serialization Model

### 6.1 Serialized Engine Loading

TensorRT uses a build-once-run-many approach:
- Models are optimized and compiled into serialized plan files (`.trt`/`.engine`)
- At runtime, the serialized engine is loaded via `IRuntime::deserializeCudaEngine()`
- Engine files are architecture-specific and cannot be shared across different GPU models

### 6.2 Loading Characteristics

- **Single sequential read**: engine file read into host memory buffer
- **No mmap**: full engine loaded into CPU memory, then parsed by TensorRT runtime
- **CUDA pinned memory**: host staging buffers use `cudaHostAlloc()` for faster
  PCIe transfers
- **Lean runtime** (TensorRT 10+): reduces runtime overhead and memory footprint
- **Refittable engines**: weight updates without full rebuild

### 6.3 No I/O-Level Optimization

TensorRT does not optimize the I/O read itself — the philosophy is that engine
compilation is the expensive step (done offline), and engine loading is fast enough
with sequential read since plan files are typically well-compressed and smaller than
original model weights.

---

## 7. Comparative Summary

### 7.1 I/O Strategy Comparison Table

| Feature | llama.cpp | safetensors | PyTorch | vLLM | ONNX RT | TensorRT | **OpenVINO** |
|---------|-----------|-------------|---------|------|---------|----------|-------------|
| **mmap** | ✅ | ✅ | ✅ (2.1+) | Via safetensors | ❌ | ❌ | ✅ (SharedStreamBuffer) |
| **O_DIRECT** | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Parallel pread** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (ParallelReadStreamBuf) |
| **Async GPU upload** | ✅ (4×64MB) | ❌ | ❌ | ✅ (tensorizer) | ❌ | ❌ | ❌ |
| **Zero-copy format** | ✅ (GGUF) | ✅ | ❌ (pickle) | ✅ | ✅ (ORT) | ✅ (plan) | ✅ (OV blob) |
| **Per-fd readahead** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| **Multi-file split** | ✅ | ✅ (sharded) | ❌ | ✅ (TP shards) | ✅ (external) | ❌ | ❌ |
| **NUMA-aware** | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |

### 7.2 Bandwidth Optimization Techniques Taxonomy

**Level 1: Format Design** (reduce bytes to read)
- Contiguous tensor layout (safetensors, GGUF)
- FlatBuffers zero-copy (ORT format)
- Pre-compiled/optimized format (TensorRT plan, OV blob)
- Split/sharded files for selective loading (safetensors, GGUF, vLLM TP shards)

**Level 2: User-space I/O** (maximize read throughput from SSD)
- mmap for zero-copy CPU access (llama.cpp, safetensors, PyTorch, OpenVINO)
- O_DIRECT for page cache bypass (llama.cpp)
- Parallel pread with per-thread fds (OpenVINO)
- Sequential full-file read (ONNX RT, TensorRT)

**Level 3: SSD→GPU Transfer** (pipeline disk→host→device)
- Async multi-buffer GPU upload with pinned memory (llama.cpp)
- Direct-to-GPU deserialization (vLLM tensorizer)
- Synchronous CPU→GPU copy (PyTorch, ONNX RT)

**Level 4: Memory Reduction** (reduce peak memory)
- Meta device model initialization (PyTorch, vLLM)
- assign=True load_state_dict (PyTorch)
- ORT model bytes for initializers (ONNX RT)
- mmap unmap_fragment (llama.cpp)
- mlock for working set pinning (llama.cpp)

### 7.3 Key Insights

1. **llama.cpp is the most I/O-sophisticated**: It is the only framework that
   implements O_DIRECT + async GPU upload pipeline, making it the best for
   maximizing raw SSD bandwidth to GPU. The 4×64MB staging buffer design
   effectively pipelines NVMe reads with PCIe transfers.

2. **safetensors wins on simplicity**: By making the file format itself efficient
   (contiguous tensors, tiny header), it achieves good performance with minimal
   code. The format design makes mmap effective without needing complex I/O logic.

3. **OpenVINO's parallel pread is unique**: No other framework uses multi-threaded
   pread for model loading, and the per-thread fd approach to avoid readahead
   interference is novel. This fills a gap between mmap (which creates PTEs in RSS)
   and single-threaded read (which under-utilizes NVMe bandwidth).

4. **vLLM's tensorizer provides direct-to-GPU**: By deserializing directly into GPU
   memory, tensorizer avoids the host staging step entirely for CUDA devices.

5. **ONNX Runtime and TensorRT are weakest on I/O**: Both rely on simple sequential
   file reads. ONNX RT compensates with the ORT format's zero-copy initialization.
   TensorRT compensates by having smaller engine files (post-optimization).

6. **No framework combines all techniques**: The ideal system would combine O_DIRECT
   (llama.cpp) + parallel pread (OpenVINO) + async GPU upload (llama.cpp) +
   format-level zero-copy (safetensors). OpenVINO's parallel pread could be
   enhanced with O_DIRECT support and async GPU upload for even better throughput.

---

## 8. Recommendations for OpenVINO

Based on the competitive analysis, potential improvements to OpenVINO's I/O pipeline:

1. **O_DIRECT support**: Like llama.cpp, bypass page cache when loading into GPU memory.
   This avoids polluting the page cache with model data that will not be re-accessed,
   and can improve throughput on NVMe SSDs.

2. **Async GPU upload pipeline**: Implement a multi-buffer pinned memory staging
   approach similar to llama.cpp's 4×64MB design. While CPU reads into buffer N,
   GPU processes buffer N-1.

3. **NUMA-aware allocation**: For multi-socket CPU inference, allocate model data
   on the local NUMA node to avoid remote memory access penalties.

4. **Format-level optimization**: Ensure the OpenVINO blob format stores tensors
   contiguously and supports selective/lazy loading for individual layers.

5. **GDS (GPU Direct Storage) integration**: For NVMe→GPU without CPU bounce,
   NVIDIA's GPUDirect Storage could provide additional bandwidth. This would
   complement or replace the async GPU upload approach.

---

## 9. Theoretical Maximum Model Loading Speed Analysis (GPU Inference)

### 9.1 Test Assumptions

- **Model size**: 10 GB
- **Storage**: PCIe 3.0 x4 NVMe SSD (rated **4 GB/s** sequential read) — typical
  mainstream NVMe (e.g., Samsung 970 EVO Plus, WD SN750, Intel 670p class)
- **GPU link**: PCIe 4.0 x16 (~25 GB/s practical)
- **Memory**: DDR4/DDR5 dual-channel (~40–80 GB/s)
- **OS**: Linux 6.x and Windows 10/11
- **Scenario**: GPU inference only (dGPU and Intel iGPU)

### 9.2 NVMe Queue Depth vs Effective Bandwidth (4 GB/s SSD)

NVMe SSDs require high queue depth (QD) to approach rated bandwidth:

| Queue Depth | Effective BW (4 GB/s SSD) | Achieved By |
|-------------|--------------------------|-------------|
| QD=1 | ~1.2–1.8 GB/s | Single-thread `read()` / `ReadFile()` |
| QD=2–4 | ~2.0–3.0 GB/s | mmap with OS readahead |
| QD=4–8 | ~3.0–3.8 GB/s | Multi-thread pread / ReadFile (few threads) |
| QD=16–32 | ~3.6–4.0 GB/s | OpenVINO 16–32 thread pread / approaching SSD ceiling |
| QD=32+ | ~4.0 GB/s | SSD hardware limit reached |

With a 4 GB/s SSD, the gap between single-thread and optimal QD is **~2–3×** (vs ~4×
on a 7 GB/s drive). Parallel I/O techniques still matter significantly.

### 9.3 I/O Mechanism Comparison: Linux vs Windows

| Mechanism | Linux | Windows | Effective BW (4 GB/s SSD) |
|-----------|-------|---------|--------------------------|
| **Single-thread buffered read** | `read()` | `ReadFile()` | 1.2–1.8 GB/s |
| **mmap** | `mmap()` + page faults | `CreateFileMapping()`+`MapViewOfFile()` | 1.5–2.5 GB/s |
| **Multi-thread positional read (16–32 threads)** | `pread()` per-thread fd | `ReadFile()` + `OVERLAPPED` per-thread handle | 3.5–4.0 GB/s |
| **O_DIRECT / unbuffered** | `open(O_DIRECT)` | `CreateFileW(FILE_FLAG_NO_BUFFERING)` | 3.5–4.0 GB/s |
| **GDS** | NVIDIA `cuFile` | NVIDIA `cuFile` (Win support limited) | 3.5–4.0 GB/s |

#### Platform-Specific Notes

**Linux:**
- `pread()` is truly positional — does not affect fd's file offset, safe for concurrent use
- Per-thread fd avoids `file_ra_state` readahead interference (OpenVINO's key innovation)
- `O_DIRECT` requires buffer, offset, and length aligned to filesystem block size (typically 512 or 4096)
- `io_uring` can further improve QD utilization but is not used by any framework surveyed

**Windows:**
- `ReadFile()` with `OVERLAPPED` structure provides positional read equivalent to `pread()`
- Per-thread `HANDLE` via `CreateFileW()` achieves identical readahead isolation as per-thread fd on Linux
- `FILE_FLAG_NO_BUFFERING` is the Windows equivalent of `O_DIRECT` — same alignment constraints
- `FILE_FLAG_SEQUENTIAL_SCAN` hints the OS to optimize readahead (useful for mmap/sequential paths)
- Windows memory-mapped I/O via `MapViewOfFile` tends to have higher per-fault overhead than Linux mmap
- Windows Prefetcher / SuperFetch may partially mitigate cold-read penalties for frequently-loaded models

### 9.4 Per-Framework GPU Loading Time Estimates (10 GB Model, 4 GB/s SSD)

#### dGPU Scenario (NVIDIA / Intel Arc — discrete GPU with PCIe link)

Data path: `NVMe → DRAM (host) → PCIe → GPU VRAM`

GPU transfer time (10 GB over PCIe 4.0 x16): ~0.4 s

| Framework | I/O Method | Linux BW | Windows BW | Linux Time | Windows Time | Bottleneck |
|-----------|-----------|---------|-----------|-----------|-------------|-----------|
| **llama.cpp** (O_DIRECT + async) | O_DIRECT + 4×64MB pipeline | 3.5–4.0 GB/s | 3.5–4.0 GB/s¹ | **~2.5 s** | **~2.5 s** | SSD ceiling, I/O+GPU overlap |
| **vLLM** (tensorizer + GDS) | GPUDirect Storage | 3.5–4.0 GB/s | N/A² | **~2.5 s** | N/A | SSD ceiling, NVMe→GPU DMA |
| **OpenVINO** (ParallelReadStreamBuf) | 16–32 thread pread + sync GPU copy | 3.5–4.0 GB/s | 3.5–4.0 GB/s | **~2.5–3.3 s** | **~2.5–3.3 s** | 16–32 thread read + 0.4 s GPU copy |
| **vLLM** (tensorizer) | Single-stream direct-to-GPU | 1.5–2.5 GB/s | N/A² | ~4.4–7.1 s | N/A | Single-stream QD=1 |
| **OpenVINO** (original SharedStreamBuffer) | mmap + memcpy + sync GPU copy | 1.5–2.5 GB/s | 1.2–2.0 GB/s | ~4.4–7.1 s | ~5.4–8.7 s | Page faults + 0.4 s GPU copy |
| **safetensors** (mmap) | mmap + sync GPU copy | 1.5–2.5 GB/s | 1.2–2.0 GB/s | ~4.4–7.1 s | ~5.4–8.7 s | mmap page faults |
| **TensorRT** | Sequential ReadFile engine | 1.2–1.8 GB/s | 1.2–1.8 GB/s | ~5.9–8.7 s | ~5.9–8.7 s | Single-thread + engine deser. |
| **ONNX Runtime** (ORT format) | ifstream::read + FlatBuffers | 1.2–1.8 GB/s | 1.2–1.8 GB/s | ~5.9–8.7 s | ~5.9–8.7 s | Single-thread QD=1 |

¹ llama.cpp uses `FILE_FLAG_NO_BUFFERING` on Windows, achieving similar unbuffered I/O.
² vLLM/tensorizer is Linux/CUDA only; GDS has limited Windows support.

#### Intel iGPU Scenario (Shared Memory — No PCIe Transfer)

Data path: `NVMe → DRAM` (iGPU accesses same DRAM, zero-copy)

GPU transfer time: **0 s** (shared memory architecture)

| Framework | I/O Method | Linux BW | Windows BW | Linux Time | Windows Time | Notes |
|-----------|-----------|---------|-----------|-----------|-------------|-------|
| **OpenVINO** (ParallelRead + O_DIRECT) | O_DIRECT pread → shared DRAM | 3.5–4.0 GB/s | 3.5–4.0 GB/s | **~2.5–2.9 s** | **~2.5–2.9 s** | Theoretical; not yet impl. |
| **OpenVINO** (ParallelReadStreamBuf) | 16–32 thread pread → shared DRAM | 3.5–4.0 GB/s | 3.5–4.0 GB/s | **~2.5–2.9 s** | **~2.5–2.9 s** | Zero-copy iGPU access |
| **OpenVINO** (original SharedStreamBuffer) | mmap → shared DRAM | 1.5–2.5 GB/s | 1.2–2.0 GB/s | ~4.0–6.7 s | ~5.0–8.3 s | Page faults limit BW |

> **iGPU advantage**: Because there is no PCIe transfer step, the total load time
> equals the pure I/O read time. This makes I/O optimization (parallel pread, O_DIRECT)
> especially impactful — every GB/s gained in read bandwidth directly reduces wall time.
> On iGPU, `O_DIRECT` is also extra valuable because it saves shared DRAM bandwidth
> that would otherwise be consumed by page-cache double-copy.

### 9.5 Key Data Points

#### mmap Page Fault Overhead (Linux vs Windows)

```
                                    Linux               Windows
Single page fault latency:         ~2–5 μs             ~3–8 μs (higher overhead)
4 KiB pages for 10 GB:             ~2,621,440 faults   ~2,621,440 faults
Raw page fault overhead:           ~5–13 s             ~8–21 s
With OS readahead (128–256 KiB):   ~40K–80K faults     ~40K–80K faults
Effective mmap bandwidth:          ~1.5–2.5 GB/s       ~1.2–2.0 GB/s
```

Windows `MapViewOfFile` tends to be ~15–30% slower than Linux `mmap` for sequential
first-touch access due to higher per-fault kernel overhead and less aggressive
readahead in the Windows memory manager.

#### O_DIRECT / FILE_FLAG_NO_BUFFERING Advantage

```
Linux O_DIRECT (64 MB blocks):             Bypasses page cache, NVMe DMA to user buf
Windows FILE_FLAG_NO_BUFFERING (64 MB):    Identical behavior, same alignment rules
Effective BW on 4 GB/s SSD:               3.5–4.0 GB/s (near SSD ceiling)
vs buffered single-thread read:            1.2–1.8 GB/s
Speedup:                                   ~2–3×
```

Both platforms require identical alignment constraints: buffer address, read offset,
and read length must be multiples of the volume's sector size (512 or 4096 bytes).

#### Async dGPU Pipeline Benefit (llama.cpp Model, 4 GB/s SSD)

```
Without pipeline:  read(10 GB) → cudaMemcpy(10 GB) = 2.5 s + 0.4 s = 2.9 s (serial)
With pipeline:     read and cudaMemcpyAsync overlap → max(2.5 s, 0.4 s) ≈ 2.5 s
Speedup:           ~14% time reduction (less than on faster SSDs because I/O dominates)
```

On a 4 GB/s SSD the I/O phase dominates (2.5 s vs 0.4 s GPU transfer), so async
pipeline provides less relative benefit compared to faster NVMe drives. The primary
optimization lever remains **maximizing SSD read bandwidth**.

### 9.6 OpenVINO Before vs After — ParallelReadStreamBuf Impact (GPU Inference)

#### dGPU Scenario (10 GB, 4 GB/s SSD)

| Metric | Original (SharedStreamBuffer) | With ParallelReadStreamBuf | Improvement |
|--------|-------------------------------|---------------------------|-------------|
| I/O mechanism | mmap + single-thread memcpy | 16–32 thread pread/ReadFile, per-thread fd/HANDLE | Fundamental change |
| Effective NVMe QD | QD=1–4 (readahead dependent) | QD=16–32 (one per thread) | 4–32× QD |
| Effective read BW (Linux) | 1.5–2.5 GB/s | 3.5–4.0 GB/s | **~1.6–2.7× BW** |
| Effective read BW (Windows) | 1.2–2.0 GB/s | 3.5–4.0 GB/s | **~2–3.3× BW** |
| 10 GB load time (Linux, dGPU) | 4.4–7.1 s | 2.5–3.3 s | **~1.7–2.2× faster** |
| 10 GB load time (Windows, dGPU) | 5.4–8.7 s | 2.5–3.3 s | **~2.2–2.6× faster** |
| RSS impact | mmap creates PTEs (in RSS) | pread uses page cache (not in RSS) | ~50% RSS reduction |

> **Windows sees larger relative improvement** because the baseline mmap (`MapViewOfFile`)
> performance is worse on Windows than on Linux, while multi-thread `ReadFile` with
> `OVERLAPPED` performs equivalently to Linux `pread`.

#### Intel iGPU Scenario (10 GB, 4 GB/s SSD)

| Metric | Original (SharedStreamBuffer) | With ParallelReadStreamBuf | Improvement |
|--------|-------------------------------|---------------------------|-------------|
| Effective read BW (Linux) | 1.5–2.5 GB/s | 3.5–4.0 GB/s | ~1.6–2.7× BW |
| Effective read BW (Windows) | 1.2–2.0 GB/s | 3.5–4.0 GB/s | ~2–3.3× BW |
| GPU transfer overhead | 0 s (shared DRAM) | 0 s (shared DRAM) | Same |
| 10 GB load time (Linux, iGPU) | 4.0–6.7 s | 2.5–2.9 s | **~1.6–2.3× faster** |
| 10 GB load time (Windows, iGPU) | 5.0–8.3 s | 2.5–2.9 s | **~2–2.9× faster** |

#### Potential Further Improvement with O_DIRECT / FILE_FLAG_NO_BUFFERING

| Metric | ParallelReadStreamBuf | + O_DIRECT (theoretical) | Improvement |
|--------|----------------------|--------------------------|-------------|
| Effective read BW | 3.5–4.0 GB/s | 3.8–4.0 GB/s | +0–15% |
| Page cache pollution | Yes | **No** | Eliminates cache pressure |
| DRAM BW usage | 2× (SSD→cache + cache→user) | 1× (SSD→user direct) | 50% DRAM BW saved |
| 10 GB load time (dGPU) | 2.5–3.3 s | **2.5–3.0 s** | ~0–10% faster |
| 10 GB load time (iGPU, Linux) | 2.5–2.9 s | **2.5–2.6 s** | ~0–12% faster |
| 10 GB load time (iGPU, Windows) | 2.5–2.9 s | **2.5–2.6 s** | ~0–12% faster |

> On a 4 GB/s SSD the absolute gain from O_DIRECT is smaller than on a 7 GB/s SSD,
> because the SSD itself becomes the ceiling. The key remaining benefit is eliminating
> page cache pollution and saving shared DRAM bandwidth — especially impactful on iGPU.

### 9.7 Overall Ranking — GPU Inference (10 GB Model, 4 GB/s SSD)

#### dGPU — Linux

| Rank | Framework | Time | Key Advantage |
|------|-----------|------|---------------|
| 1 | llama.cpp (O_DIRECT + async) | **~2.5 s** | SSD ceiling, I/O+GPU overlap |
| 2 | vLLM (tensorizer + GDS) | **~2.5 s** | NVMe DMA to GPU |
| 3 | **OpenVINO** (ParallelReadStreamBuf) | ~2.5–3.3 s | 16–32 thread pread + sync GPU copy |
| 4 | vLLM (tensorizer) | ~4.4–7.1 s | Single-stream, direct-to-GPU |
| 5 | **OpenVINO** (original SharedStreamBuffer) | ~4.4–7.1 s | mmap slow read + sync GPU copy |
| 6 | TensorRT | ~5.9–8.7 s | Single-thread + engine deser. |
| 7 | ONNX Runtime (ORT) | ~5.9–8.7 s | Single-thread ifstream |

#### dGPU — Windows

| Rank | Framework | Time | Key Advantage |
|------|-----------|------|---------------|
| 1 | llama.cpp (NO_BUFFERING + async) | **~2.5 s** | Unbuffered I/O + async GPU pipeline |
| 2 | **OpenVINO** (ParallelReadStreamBuf) | ~2.5–3.3 s | 16–32 thread ReadFile + OVERLAPPED |
| 3 | **OpenVINO** (original SharedStreamBuffer) | ~5.4–8.7 s | MapViewOfFile slow + sync GPU copy |
| 4 | TensorRT | ~5.9–8.7 s | Single-thread ReadFile |
| 5 | ONNX Runtime (ORT) | ~5.9–8.7 s | Single-thread ifstream |

> vLLM and GDS are not available on Windows, making llama.cpp and OpenVINO the primary
> contenders. OpenVINO's ParallelReadStreamBuf provides the best available framework-level
> optimization on Windows.

#### Intel iGPU — Linux

| Rank | Framework | Time | Key Advantage |
|------|-----------|------|---------------|
| 1 | **OpenVINO** (ParallelRead + O_DIRECT) | **~2.5–2.9 s** | O_DIRECT + zero-copy (theoretical) |
| 2 | **OpenVINO** (ParallelReadStreamBuf) | ~2.5–2.9 s | 16–32 thread pread + zero-copy |
| 3 | **OpenVINO** (original SharedStreamBuffer) | ~4.0–6.7 s | mmap + zero-copy but slow read |

#### Intel iGPU — Windows

| Rank | Framework | Time | Key Advantage |
|------|-----------|------|---------------|
| 1 | **OpenVINO** (ParallelRead + NO_BUFFERING) | **~2.5–2.9 s** | Unbuffered + zero-copy (theoretical) |
| 2 | **OpenVINO** (ParallelReadStreamBuf) | ~2.5–2.9 s | 16–32 thread ReadFile + zero-copy |
| 3 | **OpenVINO** (original SharedStreamBuffer) | ~5.0–8.3 s | MapViewOfFile + zero-copy but slow |

> **Key takeaway**: On a mainstream 4 GB/s NVMe SSD, OpenVINO's `ParallelReadStreamBuf`
> with 16–32 threads essentially saturates the SSD hardware ceiling. The remaining
> headroom for O_DIRECT/`FILE_FLAG_NO_BUFFERING` is minimal (~0–12%). The largest wins
> come from moving away from single-thread mmap, which is especially slow on Windows.
