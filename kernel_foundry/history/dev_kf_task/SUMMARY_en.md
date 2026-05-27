# kernel_foundry Project: Common Issues & Solutions

## 1. Celery Queue Initialization Errors
- **Issue**: Celery was initialized even when `use_queue=false`, causing env/SSL errors.
- **Solution**: Refactored `TaskRunner` to lazy-load Celery only if `use_queue=true`. Added regression test to prevent future breakage.

## 2. pytest Path & Import Failures
- **Issue**: Running pytest directly in workspace failed to import `kernelfoundry` fixtures.
- **Solution**: Workspace `conftest.py` injects `kernelfoundry.internal/kernelfoundry` into `sys.path` at runtime.

## 3. OpenCL Kernel Launch Mismatch
- **Issue**: Kernel evolution (e.g., DPAS/reqd_work_group_size/tile macros) made launch config in `task.py` outdated, causing launch failures or zeroed results.
- **Solution**: `task.py` now parses `reqd_work_group_size` and tile macros from kernel source to auto-derive launch parameters. If parsing fails, falls back with a warning.

## 4. Dtype Mismatch Causing Incorrect Results
- **Issue**: Mixing float16/float32 for input/output matrices led to large numerical errors.
- **Solution**: Use float16 for both input and output; cast to float32 for comparison, with relaxed tolerance for DPAS precision.

## 5. Kernel Compilation Failure Fallback
- **Issue**: Auto-generated kernels may fail to compile (e.g., intrinsic signature mismatch).
- **Solution**: On build failure, `task.py` falls back to reference kernel to keep tests running.

## 6. Log/Output Confusion
- **Issue**: Terminal output from multiple runs was mixed, making it hard to identify the latest results.
- **Solution**: Recommend clearing terminal before each test, or using structured logs to distinguish runs.

## 7. Lessons Learned
- When kernel/task are decoupled, always auto-sync launch metadata to avoid manual drift.
- Test harnesses should be robust to kernel evolution: prefer auto-derivation, fallback, and warnings.
- Regression tests and path/import bootstrapping are essential for reliability.

## 7. KernelFoundry/DPAS/Server-side Typical Issues
- **1)** DPAS kernel compilation failures are solved; root cause was task wrapping/launch details, not DPAS itself.
- **2)** KernelFoundry optimization is slow (tens of minutes per kernel), and generated kernel performance is often worse than copilot's 10-minute result.
- **3)** Enabling evolutionary algorithms makes generation even slower (hours), with little performance gain; EA is not as effective as expected.
- **4)** Debugging with remote KernelFoundry service is hard; local deployment is strongly recommended for troubleshooting.
- **5)** GNAI token consumption is extremely high and can be exhausted quickly; long wait times (e.g., 77000 seconds) are common when quota is exceeded.

---
See SUMMARY_cn.md for the Chinese version.
