# ParallelMemStreamBuf vs SharedStreamBuffer — Memory Consumption Analysis

## Context

At [plugin.cpp:463](../../src/plugins/intel_gpu/src/plugin/plugin.cpp), the GPU plugin's
`import_model(const ov::Tensor&, ...)` overload creates a `ParallelMemStreamBuf` to
deserialize the cache blob:

```cpp
ov::util::ParallelMemStreamBuf par_buf(model.data(), model.get_byte_size());
std::istream stream(&par_buf);
return import_model(stream, context, config);
```

This document analyzes the memory impact compared to the original `SharedStreamBuffer`
approach, with particular attention to scenarios where the mmap region is referenced by
other parts of OpenVINO.

---

## 1. Per-Object Overhead Comparison

| | `SharedStreamBuffer` | `ParallelMemStreamBuf` |
|---|---|---|
| Stack/heap size | ~24 bytes (3 pointers/size_t) | ~40 bytes + `unique_ptr<ParallelReadStreamBuf>` |
| `ParallelReadStreamBuf` | None | ~8 KB underflow buffer + fd/HANDLE + path string |
| Construction-time I/O | None | Linux: reads `/proc/self/maps`; Win32: `VirtualQuery` + `GetMappedFileNameW` |

The per-object overhead difference is negligible (a few KB vs a few tens of bytes) for
GB-scale model blobs.

---

## 2. Core Difference: Whether mmap Pages Are Faulted In

This is the critical distinction. Tracing the full call chain:

### Step 1 — Header Parsing Stage (core_impl.cpp:1574)

```cpp
header.read_from_buffer(
    static_cast<const char*>(tensor.data()),  // ← directly reads mmap memory
    tensor.get_byte_size(),
    compiled_blob_offset);
```

This code **directly accesses mmap memory** to parse the header XML, faulting in the
first few pages (typically < 4 KB). **Both approaches incur this cost equally** since
header parsing occurs before `ParallelMemStreamBuf` is created.

### Step 2 — Model Deserialization Stage (after plugin.cpp:463)

| `SharedStreamBuffer` | `ParallelMemStreamBuf` (file-backed mmap detected) |
|---|---|
| `xsgetn()` → `memcpy(dst, m_data+offset, count)` | `xsgetn()` → `m_file_buf->sgetn(dst, n)` |
| Each memcpy accesses mmap pages → **triggers page faults** | `pread()` / `ReadFile()` reads directly from file → **no mmap page faults** |
| OS must load file pages into page cache, map into process PTEs | OS reads from page cache (or disk) into user-space dst; mmap pages remain non-resident |

---

## 3. Physical Memory Impact: Three Scenarios

### Scenario A: mmap Region Not Referenced Elsewhere (typical GPU path)

```
                    SharedStreamBuffer                  ParallelMemStreamBuf
                    ──────────────────                  ─────────────────────
mmap VA reservation  ✅ entire blob                      ✅ entire blob (same)
mmap pages resident  ✅ 100% faulted in (N GB)           ❌ ≈0 (only header, a few KB)
dst buffer (plugin)  ✅ N GB (deserialization target)    ✅ N GB (same)
pread fd overhead    None                                N temporary fds (lifetime = one read)
──────────────────────────────────────────────────────────────────────────────
Peak physical RAM    ≈ 2× blob_size                      ≈ 1× blob_size
```

**Conclusion**: `ParallelMemStreamBuf` **saves** ~50% physical RAM.

### Scenario B: mmap Region Referenced and Accessed by Other Code Concurrently

For example, if another thread/component holds a `shared_ptr` to the same
`MappedMemory` and is reading its data. (In practice, the CPU plugin uses
`SharedBuffer<ov::Tensor>` with a separate tensor instance — there is no sharing of
the same mmap tensor between plugins in current OpenVINO code.)

Theoretically:

```
                    SharedStreamBuffer                  ParallelMemStreamBuf
                    ──────────────────                  ─────────────────────
Other code faults    M pages                            M pages
This import reads    faults remaining (N-M) pages       pread reads N bytes (no mmap faults)
──────────────────────────────────────────────────────────────────────────────
mmap pages resident  N pages (all)                       M pages (only what others touched)
dst buffer           N GB                                N GB
──────────────────────────────────────────────────────────────────────────────
Peak physical RAM    M + N + dst                          M + dst
```

Even when the mmap is partially referenced, `ParallelMemStreamBuf` **does not
exacerbate** memory pressure — it avoids triggering its own page faults. The M pages
faulted in by other code are an unavoidable cost independent of this choice.

### Scenario C: Header Pages (first few pages) Faulted by read_from_buffer

At `core_impl.cpp:1574`:

```cpp
header.read_from_buffer(static_cast<const char*>(tensor.data()), ...)
```

This faults in the pages containing the header (typically 1-2 × 4 KB pages). **Both
approaches incur this cost equally** since header parsing precedes streambuf creation.
`ParallelMemStreamBuf` does **not** add any extra cost here.

---

## 4. Additional Costs Introduced by ParallelMemStreamBuf

| Additional Cost | Size | Duration |
|---|---|---|
| Construction: parsing `/proc/self/maps` | 1 I/O (~few μs) | Constructor only |
| `ParallelReadStreamBuf` underflow buffer | 8 KB | Object lifetime |
| `m_fd` (main file descriptor) | 1 fd | Object lifetime |
| Per-thread temporary fds during parallel read | N fds + N × ~8 MB thread stacks | Single `xsgetn()` call |
| `m_begin`/`m_end`/`m_current` pointers (stored but never dereferenced when `m_file_buf` is active) | 24 bytes | Object lifetime |

These costs total well under a few MB (excluding temporary thread stacks), **far less
than** the GB-scale physical RAM savings from avoiding mmap page fault-in.

---

## 5. Subtle but Harmless: Unused mmap Pointer Storage

`ParallelMemStreamBuf` always stores `m_begin`/`m_end`/`m_current` pointing into the
mmap region, but when `m_file_buf` is active, these pointers are **never dereferenced**.
All `xsgetn`/`underflow`/`uflow`/`seekoff`/`seekpos` check `if (m_file_buf)` first
and delegate entirely. Therefore these pointers do not trigger page faults — only 24
bytes of stack space are wasted.

---

## 6. Summary Table

| Dimension | `SharedStreamBuffer` | `ParallelMemStreamBuf` |
|---|---|---|
| **Peak physical RAM** (file-backed mmap) | **2× blob** (mmap WS + dst) | **1× blob** (dst only) |
| mmap region referenced elsewhere | No effect (this path already faults in everything) | **Does not exacerbate** (only others' faulted pages are resident) |
| Per-object overhead | ~24 bytes | ~8 KB + fd |
| I/O throughput | Single-threaded memcpy + serialized page faults | Multi-threaded pread, saturates NVMe bandwidth |
| Temporary parallel-read thread stacks | None | N × ~8 MB (only during large reads) |

---

## 7. Conclusion (Single-Process)

`ParallelMemStreamBuf` does **not** cause additional memory consumption compared to
`SharedStreamBuffer`. On the contrary, for file-backed mmap scenarios it **substantially
reduces** peak physical memory (from 2× blob to 1× blob) by bypassing mmap page
faults entirely and reading via independent `pread`/`ReadFile` channels. **Even when the
mmap region is referenced by other parts of OpenVINO**, `ParallelMemStreamBuf` does not
worsen the situation — it simply avoids adding its own page-fault pressure on top of
whatever other code has already faulted in.

---

## 8. Multi-Process Scenario: Mixed ParallelMemStreamBuf and SharedStreamBuffer

### Background: Page Cache, Per-Process mmap Working Set, and pread

Understanding how the OS handles these three mechanisms is essential:

| I/O Path | Page Cache | Per-Process PTE / Working Set |
|---|---|---|
| **mmap page fault** (triggered by SharedStreamBuffer memcpy) | File pages loaded into page cache | Process PTEs map to page cache physical pages → **counted in process RSS** |
| **pread / ReadFile** (used by ParallelMemStreamBuf internally) | File pages also loaded into page cache | **No mmap PTEs created** for this process → **not counted in RSS** |

Both paths load file data into the OS page cache, but only mmap page faults create
per-process PTE mappings.

### Analysis: N Processes Reading the Same Cache File (blob = B GB)

#### Baseline: All Processes Use SharedStreamBuffer

```
Process 1: mmap VA(B) + PTEs → page cache ──┐
Process 2: mmap VA(B) + PTEs → page cache ──┤  shared physical pages
...                                          │
Process N: mmap VA(B) + PTEs → page cache ──┘

Page cache:              1×B  (all processes share the same physical pages)
Per-process mmap PTEs:   N sets of PTEs (kernel RAM: ~8 bytes/page × B/4KB × N)
Per-process dst buffers: N×B  (each process's deserialization target)
────────────────────────────────────────────────────────────────────
Unique physical RAM:     1×B (page cache) + N×B (dst) = (N+1)×B
Kernel PTE overhead:     N × B/4KB × 8 bytes ≈ N × B/512
```

Page cache pages are referenced by all N processes' mmap PTEs, so they **cannot be
reclaimed** by the OS until all processes unmap.

#### Mixed: K Processes Use ParallelMemStreamBuf, (N-K) Use SharedStreamBuffer

```
ParallelMemStreamBuf processes (×K):
  mmap VA(B) exists, PTEs nearly empty (only header pages)
  pread → page cache → dst

SharedStreamBuffer processes (×(N-K)):
  mmap VA(B) + full PTEs → page cache → memcpy → dst

Page cache:              1×B  (pread and page faults populate the same physical pages)
Per-process mmap PTEs:   (N-K) full PTE sets + K near-empty PTE sets
Per-process dst buffers: N×B
────────────────────────────────────────────────────────────────────
Unique physical RAM:     1×B (page cache) + N×B (dst) = (N+1)×B   ← same as all-SharedStreamBuffer
Kernel PTE overhead:     (N-K) × B/512                             ← less than all-SharedStreamBuffer
```

### Comparison Summary

| Dimension | All SharedStreamBuffer | Mixed ParallelMemStreamBuf | Difference |
|---|---|---|---|
| **Unique physical pages (page cache + dst)** | (N+1)×B | (N+1)×B | **Zero difference** |
| Per-process RSS (apparent) | ~2×B per process | ParallelMemStreamBuf processes ~1×B; SharedStreamBuffer processes ~2×B | ParallelMemStreamBuf processes have lower RSS |
| Kernel PTE memory | N×B/512 | (N-K)×B/512 | Mixed uses **less** |
| Page cache reclaimability | N PTE sets pin pages → pages pinned until all unmap | Only (N-K) PTE sets pin pages → pages reclaimable sooner | Mixed is **better** |
| Windows PFN lock contention | N processes faulting simultaneously → severe contention | Only (N-K) processes fault | Mixed is **better** |

### Why pread Does Not Cause Double Page Cache Consumption

A common concern is whether `pread` and mmap page faults accessing the same file
regions cause the data to exist twice in page cache. **They do not.** The Linux page
cache is indexed by `(inode, offset)` — regardless of whether the data is accessed via
`read()`/`pread()` or via mmap page faults, the kernel uses the **same physical page**.
The same principle applies on Windows where the Cache Manager and Memory Manager share
the same physical pages for a given file section.

### Conclusion (Multi-Process)

**Mixed usage does not cause additional memory consumption compared to all-SharedStreamBuffer.**

1. **Page cache is system-global and shared** — `pread` and mmap page faults read the
   same physical page cache pages; there is no duplication.
2. **ParallelMemStreamBuf processes do not create mmap PTEs** — this actually saves
   per-process kernel PTE memory.
3. **Fewer PTE references mean page cache pages are reclaimable sooner** — when
   SharedStreamBuffer processes exit but ParallelMemStreamBuf processes are still
   running, the page cache pages are not pinned by mmap PTEs and the OS can reclaim
   them under memory pressure.
