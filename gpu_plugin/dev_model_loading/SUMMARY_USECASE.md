# ParallelReadStreamBuf / ParallelMemStreamBuf 使用场景总结

## 类说明

| 类 | 位置 | 作用 |
|---|---|---|
| `ParallelReadStreamBuf` | `src/common/util/include/openvino/util/parallel_read_streambuf.hpp` | 基于文件描述符 (`pread`/`ReadFile`) 的并行 streambuf；直接打开磁盘文件，多线程分块读取 |
| `ParallelMemStreamBuf` | `src/common/util/include/openvino/util/parallel_mem_streambuf.hpp` | 基于内存指针的 streambuf；检测指针是否来自文件 mmap，是则内部复用 `ParallelReadStreamBuf`，否则走 `madvise`/`PrefetchVirtualMemory` + memcpy fallback |

---

## 使用场景矩阵

| 场景 | `enable_mmap` | 设备支持 `caching_with_mmap` | 使用的类 | 代码位置 |
|---|---|---|---|---|
| **cache hit，文件缓存，非 mmap** | false | 无关 | `ParallelReadStreamBuf` | `cache_manager.hpp:71` |
| **cache hit，文件缓存，mmap** | true | ✅ (GPU) | `ov::read_tensor_data()` → `ov::Tensor` → plugin 内创建 `ParallelMemStreamBuf` | `cache_manager.hpp:62` + `plugin.cpp:460` |
| **cache hit，单文件存储，非 mmap** | false | 无关 | `ParallelReadStreamBuf`（带 offset） | `single_file_storage.cpp:280` |
| **cache hit，单文件存储，mmap** | true | ✅ (GPU) | `ov::Tensor` → plugin 内创建 `ParallelMemStreamBuf` | `single_file_storage.cpp:263` + `plugin.cpp:460` |
| **cache miss** | 任意 | 任意 | **不涉及**（输入为 `ov::Model&`，写操作） | — |
| `import_model(istream&, ...)` 直接调用 | 任意 | 任意 | 由调用者决定是否包装 | — |

---

## 完整调用链

### cache hit + enable_mmap=false
```
CoreImpl::compile_model_and_cache_impl()
  └─ CacheManager::read_cache_entry(path)
       └─ ParallelReadStreamBuf par_buf(path)          ← 直接 pread 并行读文件
            └─ istream stream(&par_buf)
                 └─ plugin.import_model(stream, config) ← CPU/GPU 通用 istream 路径
```

### cache hit + enable_mmap=true (GPU, 支持 caching_with_mmap)
```
CoreImpl::compile_model_and_cache_impl()
  └─ CacheManager::read_cache_entry(path)
       └─ ov::read_tensor_data(path) → ov::Tensor      ← OS mmap 整个文件
            └─ plugin.import_model(Tensor, config)      ← plugin.cpp:460
                 └─ ParallelMemStreamBuf mem_buf(data, size)
                      ├─ [Linux] 解析 /proc/self/maps 检测 file-backed mmap
                      │    └─ 检测到文件路径 → new ParallelReadStreamBuf(path)
                      │         └─ 多线程 pread 并行读文件（绕过 page cache 争用）
                      └─ [fallback] madvise(MADV_WILLNEED) + 串行 memcpy
                           └─ BinaryInputBuffer(stream, engine) → 反序列化 GPU kernel
```

### cache miss（任意配置）
```
CoreImpl::compile_model_and_cache_impl()
  └─ plugin.compile_model(ov::Model&, config)           ← 输入是 IR 图对象，非 stream
       └─ JIT 编译 (GPU: OpenCL kernel 编译)
            └─ export_model(ofstream) → 写缓存文件
                 ← ParallelReadStreamBuf / ParallelMemStreamBuf 均不涉及
```

---

## 关键判断条件（代码位置）

| 条件 | 代码位置 |
|---|---|
| `enable_mmap` 配置传播 | `CoreImpl` → `CoreConfig` → `CacheContent.m_mmap_enabled` |
| mmap 路径 gate | `core_impl.cpp:1565`: `m_mmap_enabled && device_supports_internal_property(plugin, ov::internal::caching_with_mmap)` |
| GPU 声明支持 mmap | `plugin.cpp:760`: 返回 `caching_with_mmap` |
| variant 分发 | `core_impl.cpp:1566`: `std::visit(model_importer, compiled_blob)` — index 0 = Tensor 路径, index 1 = stream 路径 |
| Linux file-backed mmap 检测 | `parallel_mem_streambuf.cpp` 匿名 namespace `get_mmap_file_info()`: 解析 `/proc/self/maps` |
| Windows file-backed mmap 检测 | `parallel_mem_streambuf.cpp` 匿名 namespace `resolve_device_path()`: `VirtualQuery` + `GetMappedFileNameW` |

---

## 优化说明

这两个类的设计目标是加速 **cache hit 路径**（第 2 次及以后加载同一模型）的 blob 读取，首次加载（cache miss）不受益。

- `ParallelReadStreamBuf`：多线程 `pread` 并行分块读文件，规避单线程 sequential read 的 I/O 瓶颈
- `ParallelMemStreamBuf`：对 file-backed mmap 指针，转为并行 `pread`（避免并发 page fault 争用）；对匿名 mmap（如 GPU 显存 copy），退化为 `madvise` 预取 + 串行 memcpy

阈值控制（`DEFAULT_THRESHOLD = 4 MB`）：小于阈值的 blob 走单线程 `single_read`，避免并行调度开销超过 I/O 收益。
