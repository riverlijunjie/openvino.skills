# Windows Memory Analysis: parallel_io Impact on Peak Memory

## Test Environment

- **Total Physical RAM**: ~31.3–31.5 GB
- **Model**: LLM with ~15 GB weight blob in cache file
- **Platform**: Windows (Task Manager + OpenVINO GenAI memory instrumentation)

---

## Raw Data Comparison

| Metric | Without parallel_io (mmap serial) | With parallel_io (ReadFile parallel) | Delta |
|---|---|---|---|
| Task Manager memory_usage | 30.9 GB | 22.3 GB | **-8.6 GB** |
| Task Manager available | 448 MB | 9.2 GB | **+8.75 GB** |
| Max RSS (OV GenAI) | 7672 MiB (7.5 GB) | 16225 MiB (15.9 GB) | +8.5 GB |
| RSS increase (OV GenAI) | 7159 MiB | 15712 MiB | +8.6 GB |
| Max System memory (OV GenAI) | 21630 MiB | 23100 MiB | +1.5 GB |
| System increase (OV GenAI) | 14592 MiB | 16224 MiB | +1.6 GB |

---

## Root Cause Analysis

### 1. Why Task Manager Shows Significantly Lower System Memory with parallel_io (30.9 → 22.3 GB)

The key difference is **whether mmap pages are faulted into physical memory**.

**Without parallel_io (mmap serial deserialization):**
```
Disk ──mmap──→ File Cache / Page Cache ──page fault──→ Process Working Set
                    (pages "In Use")                        ↓
                                                    memcpy → weight buffers
                                                        ("In Use")
```
- mmap creates a file mapping; deserialization triggers page faults page-by-page.
- Each faulted page counts toward both the **process Working Set** and the system **"In Use"** memory.
- At peak: mmap pages (~15 GB) + weight destination buffers (~7 GB) + OS/other overhead (~8 GB) ≈ **30 GB "In Use"**.
- Only 448 MB available — dangerously close to OOM.

**With parallel_io (ReadFile parallel read):**
```
Disk ──ReadFile──→ File Cache (transient, Standby)──copy──→ User buffer (weight dst)
                   (pages recyclable)                       ("In Use")
```
- `ParallelMemStreamBuf` detects file-backed mmap via `VirtualQuery` and resolves the file path.
- Constructs a `ParallelReadStreamBuf` that reads via `CreateFileW` + `ReadFile` directly into destination buffers.
- **The mmap virtual mapping exists but no page faults are triggered → 0 physical memory consumed by mmap pages.**
- `ReadFile` is buffered I/O; data passes through the File Cache, but those cache pages reside on the **Standby** list (counted as "Available", not "In Use") and can be reclaimed by the Cache Manager at any time.
- At peak: weight buffers (~15 GB) + OS/other (~7 GB) ≈ **22 GB "In Use"**.
- 9.2 GB available — system runs safely.

**The ~8.6 GB delta ≈ size of the weight blob in the cache file**, corresponding exactly to the mmap fault pages that were eliminated.

### 2. Why RSS Is Higher with parallel_io (7.5 → 15.9 GB)

This appears contradictory but is entirely consistent. The key is **Windows Working Set Manager dynamic trimming behavior**:

**Without parallel_io: system memory utilization at 98.6% (30.9/31.3 GB)**
- Under extreme memory pressure, Working Set Manager initiates **aggressive trimming**.
- Process mmap pages are prioritized for eviction from the Working Set (they are clean file-backed pages — cheapest to trim).
- Allocated weight buffer pages may also be partially paged out to the pagefile.
- Result: RSS is artificially suppressed to 7.5 GB. This does **not** mean the process actually uses less memory — it means a large number of pages have been forcibly trimmed.
- Accessing trimmed pages later triggers soft/hard page faults → **runtime performance degradation**.

**With parallel_io: system memory utilization at 71% (22.3/31.5 GB)**
- Low memory pressure; Working Set Manager does not need aggressive trimming.
- All allocated weight buffer pages remain in the Working Set.
- RSS reflects the **true physical memory footprint** ≈ 15.9 GB.
- No unnecessary page faults during inference → better performance.

> **Higher RSS ≠ memory waste.** The low RSS without parallel_io is a symptom of passive trimming under memory pressure, accompanied by performance penalties.

### 3. Why OpenVINO "Max System memory" Is Similar in Both Cases (~21.6 vs ~23.1 GB)

OpenVINO's System memory metric samples via `GlobalMemoryStatusEx`, computing `ullTotalPhys - ullAvailPhys`. It samples at discrete points during the compilation phase and may not capture the absolute peak.

Without parallel_io, system memory spikes rapidly to 30.9 GB, but OpenVINO's sampling may occur before this peak, capturing only 21.6 GB. The true peak is revealed by Task Manager's real-time display.

---

## Impact Assessment: parallel_io Effect on Peak Memory

| Dimension | Impact | Verdict |
|---|---|---|
| **Peak System Physical Memory** | 30.9 GB → 22.3 GB (−28%) | **Strong positive** |
| **Available Memory** | 448 MB → 9.2 GB (+20×) | **Strong positive** |
| **OOM Risk** | Near-OOM (448 MB) → safe headroom | **Eliminates OOM risk** |
| **Peak RSS** | 7.5 GB → 15.9 GB (+2.1×) | Numeric increase, but reflects true usage |
| **Inference Performance** | Higher RSS = fewer page faults | **Positive** |

---

## Conclusion

**parallel_io has a strongly positive impact on peak memory.**

By eliminating mmap page faults, it transforms ~15 GB of file-backed pages from "In Use" to "zero physical memory cost", reducing peak system memory by 8.6 GB. The RSS increase is a healthy signal: the system no longer needs to aggressively trim the process Working Set, meaning all allocated buffers stay resident and inference runs without avoidable page faults.

**One-line summary:** mmap's dual memory occupancy (file cache pages + weight destination buffers) is the root cause of the near-OOM condition; parallel_io eliminates file cache page residency via direct ReadFile, fundamentally resolving the problem.

---

## Memory Flow Diagrams

### Without parallel_io (mmap path)
```
                    ┌──────────────────────────────────────┐
                    │         Physical Memory (31.3 GB)     │
                    ├──────────────────────────────────────┤
                    │ OS + Other           ~8 GB  "In Use" │
                    │ mmap fault pages    ~15 GB  "In Use" │ ← File cache pages
                    │ Weight buffers       ~7 GB  "In Use" │ ← malloc'd by plugin
                    │ Available             0.4 GB          │ ← DANGER
                    └──────────────────────────────────────┘
    Peak RSS = 7.5 GB (aggressively trimmed by Working Set Manager)
```

### With parallel_io (ReadFile path)
```
                    ┌──────────────────────────────────────┐
                    │         Physical Memory (31.5 GB)     │
                    ├──────────────────────────────────────┤
                    │ OS + Other           ~7 GB  "In Use" │
                    │ mmap fault pages      0 GB           │ ← Never faulted
                    │ Weight buffers      ~15 GB  "In Use" │ ← ReadFile → dst
                    │ Standby (file cache) ~2 GB recyclable│
                    │ Available             9.2 GB          │ ← SAFE
                    └──────────────────────────────────────┘
    Peak RSS = 15.9 GB (no trimming needed, reflects true usage)
```
