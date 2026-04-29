# Windows GPU 模型加载优化总结 — `ENABLE_MMAP=false` 路径

## 目标

在 Windows + LNL iGPU（32GB 统一内存）上，以 `ENABLE_MMAP=false` 配置加载 qwen3-30b-a3b（16.3GB `.bin` + 15.4GB cache blob）时，消除 **第二次运行产生的 5x 性能退化** 问题，并全面提升加载速度。

---

## 问题现象

### 测试环境
- **平台**: Windows, Intel LNL iGPU, 32GB 统一内存
- **模型**: qwen3-30b-a3b, `.bin` 文件 16.3GB, cache blob 15.4GB × 2
- **配置**: `{"CACHE_DIR": "cache_dir", "ENABLE_MMAP": false}`
- **远程机器**: `Local_Admin@10.239.132.229`

### 退化数据

| 运行次数 | `read_model` (16.3GB .bin) | `import_model` (cache blob) | Pipeline 总耗时 |
|---------|---------------------------|----------------------------|----------------|
| 第 1 次 | **219,534 ms** (~75 MB/s) | 22,483 ms (~1.5 GB/s) | **243.40s** |
| 第 2 次 | **218,254 ms** (~75 MB/s) | 20,335 ms (~1.5 GB/s) | **240.13s** |

关键观察：
- `import_model`（cache blob 读取）已经使用 `ParallelReadStreamBuf`，速度正常（1.5-2 GB/s）
- `read_model`（weights .bin 读取）使用 `std::ifstream`，速度仅 ~75 MB/s
- **主要瓶颈是 `read_model` 而不是 `import_model`**
- 第 2 次运行并无额外退化——两次运行都慢（~220s），问题在 `read_model` 的绝对速度上

---

## 诊断过程

### 第一阶段：定位退化位置（误区排除）

初始假设（全部排除）：
1. ❌ Windows `FILE_FLAG_NO_BUFFERING` 导致页面缓存问题 → 改为 `SEQUENTIAL_SCAN` → 无效
2. ❌ 多线程 `ReadFile` 破坏 Windows 顺序预读检测 → 单线程 fallback → 无效
3. ❌ `OVERLAPPED` 阻止 `CurrentByteOffset` 更新 → 改用 `SetFilePointerEx+ReadFile(NULL)` → 无效
4. ❌ `std::ifstream` 退回方案 → 用户明确拒绝：「不要退回到 std::ifstream 的机制，还是需要 parallel_io 的」

### 第二阶段：添加 DIAG 诊断日志

在以下位置注入 `[DIAG]` 时间戳：

| 文件 | 诊断点 |
|------|--------|
| `core_impl.cpp` | `CoreImpl::read_model(path)` START/DONE |
| `core_impl.cpp` | `CoreImpl::compile_model(model, device)` START/DONE |
| `core_impl.cpp` | `CoreImpl::compile_model(path, device)` START/DONE |
| `core_impl.cpp` | `load_model_from_cache` START/DONE，`import_model` START/DONE |
| `parallel_read_streambuf.cpp` | 构造/析构，xsgetn 大块读取的吞吐量 |
| `cache_manager.hpp` | `read_cache_entry` START/DONE |
| `single_file_storage.cpp` | `read_cache_entry` START/DONE |

### 第三阶段：关键发现

诊断日志揭示了完整的 Pipeline 时间线：

```
[DIAG] CoreImpl::read_model START  model=openvino_tokenizer.xml     →  20 ms
[DIAG] CoreImpl::read_model START  model=openvino_detokenizer.xml   →   6 ms
[DIAG] compile_model(model) START  device=CPU                        →  79 ms (tokenizer)
[DIAG] compile_model(model) START  device=CPU                        →  17 ms (detokenizer)
[DIAG] CoreImpl::read_model START  model=openvino_model.xml          → 219,534 ms ★★★ 瓶颈！
[DIAG] compile_model(model) START  device=GPU
[DIAG]   load_model_from_cache → import_model                        →  22,483 ms (正常)
Pipeline initialization time: 243.40s
```

**219 秒中有 220 秒花在了 `read_model` 上**——而非之前怀疑的 `import_model`。

### 第四阶段：追踪 read_model 的实际 I/O 路径

调用链：
```
CoreImpl::read_model(model_path, bin_path, properties)
  → ov::util::read_model(model_path, bin_path, extensions, enable_mmap)
    → ov::frontend::FrontEnd::load_impl(params)   [IR frontend]
      → 检测 enable_mmap 参数
        → if (enable_mmap)  → mmap 映射
        → else              → std::ifstream::read() ← 瓶颈所在！
```

**根因代码** (修改前，`src/frontends/ir/src/frontend.cpp`):
```cpp
} else if (std::ifstream bin_stream(weights_path, std::ios::binary); bin_stream.is_open()) {
    bin_stream.seekg(0, std::ios::end);
    size_t file_size = bin_stream.tellg();
    bin_stream.seekg(0, std::ios::beg);

    auto aligned_weights_buffer = std::make_shared<ov::AlignedBuffer>(file_size);
    bin_stream.read(aligned_weights_buffer->get_ptr<char>(), aligned_weights_buffer->size());
    // 16.3GB 单线程顺序读 → ~75 MB/s → 220 秒
```

同一 pipeline 中的 cache blob 读取使用 `ParallelReadStreamBuf` 达到 1.5-2 GB/s，两者形成鲜明对比。

---

## 优化方案

### 修改文件

**`src/frontends/ir/src/frontend.cpp`**（IR 前端，权重加载路径）

### 修改内容

将 `enable_mmap=false` 分支的 `std::ifstream` 替换为 `ParallelReadStreamBuf`：

```cpp
// 新增 include
#include "openvino/util/parallel_read_streambuf.hpp"

// 修改前:
} else if (std::ifstream bin_stream(weights_path, std::ios::binary); bin_stream.is_open()) {
    bin_stream.seekg(0, std::ios::end);
    size_t file_size = bin_stream.tellg();
    bin_stream.seekg(0, std::ios::beg);
    auto aligned_weights_buffer = std::make_shared<ov::AlignedBuffer>(file_size);
    bin_stream.read(aligned_weights_buffer->get_ptr<char>(), aligned_weights_buffer->size());
    // ...
} else {
    OPENVINO_THROW("Weights file ", weights_path, " cannot be opened!");
}

// 修改后:
} else {
    ov::util::ParallelReadStreamBuf read_buf(weights_path, 0);
    std::istream bin_stream(&read_buf);
    bin_stream.seekg(0, std::ios::end);
    size_t file_size = bin_stream.tellg();
    bin_stream.seekg(0, std::ios::beg);
    auto aligned_weights_buffer = std::make_shared<ov::AlignedBuffer>(file_size);
    bin_stream.read(aligned_weights_buffer->get_ptr<char>(), aligned_weights_buffer->size());
    // ...
}
```

### 设计考量

1. **链接依赖**: IR frontend 链接 `openvino::core::dev`，后者 INTERFACE 链接 `openvino::util`，因此 `ParallelReadStreamBuf` 头文件和符号均可用，无需修改 CMakeLists.txt。

2. **错误处理**: `ParallelReadStreamBuf` 构造函数在文件不存在时直接抛异常，因此不再需要原来的 `bin_stream.is_open()` 检查和 `OPENVINO_THROW` fallback 分支。

3. **行为一致性**: 修改仅影响 `enable_mmap=false` 路径；`enable_mmap=true`（mmap）路径不受影响。

4. **平台通用**: `ParallelReadStreamBuf` 内部对 Linux（`pread + posix_fadvise`）和 Windows（`SetFilePointerEx + ReadFile`）都有实现，修改后两个平台都能受益。

---

## 优化效果

### Windows 实测数据（qwen3-30b-a3b, 16.3GB .bin）

| 指标 | 优化前 (std::ifstream) | 优化后 (ParallelReadStreamBuf) | 加速比 |
|------|----------------------|-------------------------------|--------|
| `read_model` 耗时 | 218,842 ms | **9,328 ms** | **23.5x** |
| `read_model` 带宽 | ~75 MB/s | **~1.7 GB/s** | — |
| Pipeline 总耗时（第 1 次运行） | ~243s | **28.22s** | **8.6x** |
| Pipeline 总耗时（第 2 次运行） | ~241s | **29.24s** | **8.2x** |
| 第 2 次运行退化 | 无（两次都慢） | ✅ 无退化 | — |

### 时间线对比

**优化前**:
```
read_model tokenizer/detokenizer:           ~0.1s
read_model openvino_model.bin (ifstream):  219.0s  ← 瓶颈
compile_model → import_model (cache):       22.0s
Pipeline total:                            ~241.0s
```

**优化后**:
```
read_model tokenizer/detokenizer:           ~0.1s
read_model openvino_model.bin (parallel):    9.3s  ← 23x 加速
compile_model → import_model (cache):       18.0s
Pipeline total:                             28.2s
```

---

## 关键教训

1. **不要只看你优化过的代码路径**：此前 `ParallelReadStreamBuf` 仅应用于 cache blob（`cache_manager.hpp` / `single_file_storage.cpp`），而忽略了同一 pipeline 中最大的 I/O 操作——`.bin` 权重文件的读取。

2. **诊断日志是定位性能问题的利器**：5 轮假设+修改的试错全部失败，一轮 DIAG 注入+实测立即定位根因。

3. **注意 DLL 部署路径**：Windows 远程测试环境中，`run_llm.bat` 的 `PYTHONPATH` 和 `PATH` 指向 `%openvino_install_dir%\release_install\...`，而 cmake `--install --prefix` 的路径必须与之匹配。此外还要注意 `py310\Lib\site-packages\openvino\libs\` 目录下的旧版 DLL 可能覆盖新构建的版本。

4. **`std::ifstream` 在大文件场景下性能极差**：16.3GB 文件单线程 `read()` 仅 ~75 MB/s（Windows），这远低于 NVMe 硬件能力。根因是单线程无法填满 NVMe 的 I/O 队列深度。

---

## 修改的文件清单

| 文件 | 修改类型 | 说明 |
|------|---------|------|
| `src/frontends/ir/src/frontend.cpp` | **核心优化** | `enable_mmap=false` 时用 `ParallelReadStreamBuf` 替换 `std::ifstream` |

注：以上是唯一的功能性修改。`core_impl.cpp`、`parallel_read_streambuf.cpp`、`cache_manager.hpp`、`single_file_storage.cpp` 中的 `[DIAG]` 日志为诊断辅助代码，需在最终提交前清除。
