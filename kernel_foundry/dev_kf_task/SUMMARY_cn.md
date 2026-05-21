# kernel_foundry 项目常见问题与解决方案总结

## 1. Celery 队列初始化异常
- **问题**：`use_queue=false` 时仍然初始化 Celery，导致环境变量/SSL 报错。
- **解决**：重构 `TaskRunner`，仅在 `use_queue=true` 时懒加载 Celery，增加回归测试防止回归。

## 2. pytest 路径与依赖导入失败
- **问题**：在 workspace 目录下直接跑 pytest，找不到 `kernelfoundry` 相关 fixture。
- **解决**：在 workspace 的 `conftest.py` 里动态插入 `kernelfoundry.internal/kernelfoundry` 到 `sys.path`。

## 3. OpenCL kernel launch 配置与 kernel 代码不一致
- **问题**：kernel 进化后（如 DPAS/reqd_work_group_size/tile 宏变更），但 `task.py` 仍用旧的 launch 规则，导致 launch 失败或结果全零。
- **解决**：`task.py` 自动解析 kernel 源码中的 `reqd_work_group_size` 和 tile 宏，动态推导 launch 参数。若解析不到，自动降级并告警。

## 4. dtype 不匹配导致结果异常
- **问题**：输入/输出矩阵类型 float16/float32 混用，导致结果极大偏差。
- **解决**：输入输出均用 float16，比较时转 float32，容忍 DPAS 精度损失。

## 5. kernel 编译失败 fallback 机制
- **问题**：自动生成的 kernel 可能因 intrinsic 签名等原因编译失败。
- **解决**：`task.py` 检测到编译失败时自动 fallback 到 reference kernel，保证测试链路不中断。

## 6. 日志与输出混乱
- **问题**：多次运行后 terminal 输出混杂，难以判断最新结果。
- **解决**：建议每次测试前清理 terminal，或用结构化日志区分本次运行。

## 7. 经验教训
- kernel/task 分离时，launch 相关元数据应自动同步，避免人工同步失配。
- 测试 harness 应对 kernel 进化具备鲁棒性，优先自动推导、降级、告警。
- 回归测试和路径/依赖自举机制必不可少。

## 7. KernelFoundry/DPAS/服务端典型问题
- **1）** DPAS kernel 不能编译问题已解决，主要是 task 包装方式导致，需注意接口和 launch 细节。
- **2）** KernelFoundry 优化 kernel 时间很长，通常几十分钟，性能也不如 copilot 10 分钟生成的 kernel。
- **3）** 启用进化算法后，生成时间更长（小时级），但 kernel 性能提升有限，进化算法效果不如预期。
- **4）** 远程 KernelFoundry 服务调试困难，建议本地搭建环境 debug。
- **5）** GNAI token 消耗极快，容易耗尽，需注意额度和排队等待（如需等待 77000 秒）。

---
如需英文版请见 SUMMARY_en.md。
