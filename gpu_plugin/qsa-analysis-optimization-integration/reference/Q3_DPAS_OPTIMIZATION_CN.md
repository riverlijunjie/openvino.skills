# Q3 DPAS 访存优化与 B580 实测

日期：2026-09-10。范围仅为 `cm_kernel/` 独立 kernel，**未修改 OpenVINO 插件中的副本**。

## 1. 保持的 QSA 语义

参考 `../QSA_GR_ANALYSIS_CN.md`：每个 token 对自己选中的完整 blocks 的原始 K/V 做 attention，
再加入当前不完整 block 的 causal tail。所有 query heads 共享该 token 的选择，但不能共享其他 token 的 mask。
保留原有 GQA tiling、QK/PV DPAS、FP32 online softmax、output gate 和 SLM barrier 协议；不改 Q2 selection。

## 2. 三项优化

三个 JIT 开关默认均为 1，全部设为 0 可恢复旧计算/加载路径用于 A/B：

1. **`QSA_COALESCE_KV`：页内连续 blocks 合并加载。**
   一个 step 的 blocks 必须全部有效、连续，而且 16 个 KV tokens 必须位于同一逻辑 page。
   满足时使用 `16 × 32 half` 的 2D load，而不是逐个 `4 × 32 half` load。
   对 `Dh=Dv=256, W=4, r=4`，每 worker 每 step 的 K/V load 从 **16 条降为 4 条**。
   每行仍不超过 64 bytes；跨页、含间隙和部分 step 回退到旧加载路径。
   不假设物理 pages 连续，也不改变每列的 selection mask。
2. **`QSA_WIDE_TAIL`：tail 使用宽 2D K/V 加载。**
   以 r 对齐的块读取，避免跨物理 page；descriptor 的有效高度限制为剩余有效行数，
   硬件为部分 VNNI row pair 补零。替代原有逐 token、逐 16-dim 的小加载与手工 VNNI 排布。
3. **`QSA_CACHE_HEADS`：缓存归并游标头。**
   只在对应 token 的游标前进时重新读取 selected block，不重复加载未变化的头。
   对独立、稀疏的选择更有效；不增加 barrier 或 kernel 参数。

新增静态断言：page size 必须为 r 的倍数，worker 的 K/V slice 必须可被 32 half 的宽加载完整覆盖。

## 3. 测试环境与方法

- 远程：`remote_machine.txt` 指定的 Intel Arc B580，20 Xe cores。
- 目录：`/mnt/river/qsa/cm_kernel`；Python：`.venv/bin/python`。
- 编译器：`CM_FE_DIR=/mnt/river`，`-cmc -Qxcm_register_file_size=256`。
- 配置：`Hq=24, Hkv=2, Dh=Dv=256, r=4, PA_BLOCK=16, W=4, HT=4`。
- 同一随机输入、同一设备输入 buffers，打乱物理 page table；selection 由 harness 合成，并非模型真实 Q2 输出。
- 每个变体 100 次 warmup；2 轮、每轮 30 个计时样本；第二轮倒转变体顺序。
- 每个计时样本前单独冲刷 96 MiB 缓存；使用 GPU event duration，不计编译、上传、CPU reference 或 flush 时间。
- 下表为两轮共 60 个样本的平均延迟；不是端到端模型加速比。

## 4. 结果

| block_topk | prefill tokens | 原始 ms | 优化 ms | 加速比 |
|---:|---:|---:|---:|---:|
| 512 | 1024 | 1.071510 | 0.882607 | 1.214× |
| 512 | 2048 | 3.801710 | 3.157680 | 1.204× |
| 512 | 4096 | 14.068746 | 12.301930 | 1.144× |
| 512 | 8192 | 47.046783 | 41.746708 | 1.127× |
| 4 | 1024 | 0.260406 | 0.212501 | 1.225× |
| 4 | 4096 | 1.046454 | 0.855654 | 1.223× |

所有上述 A/B 输出逐元素相同。第一次短序列单进程测试存在明显频率爬升波动，故不采用最初单次 sweep 的均值作为最终比较数据。

### 消融：默认 top-k=512

| tokens | 原始 | 仅合并加载 | 再加宽 tail | 再加游标头缓存（默认） |
|---:|---:|---:|---:|---:|
| 1024 | 1.071510 | 0.869621 | 0.857835 | 0.882607 |
| 2048 | 3.801710 | 3.040465 | 3.045430 | 3.157680 |
| 4096 | 14.068746 | 12.485635 | 12.603920 | 12.301930 |

不是每一项优化在所有 shape 上都独立获益：密集前缀中游标通常同步前进，缓存头不减少读取，
1024/2048 tokens 时反而略慢；稀疏场景与 4096-token 默认场景受益。默认统一开启三项，避免按 benchmark shape 特化。
如部署端已知工作负载主要是短 dense prefix，可单独评估 `QSA_CACHE_HEADS=0`。

top-k=4、4096 tokens 的消融为：1.046454 → 1.046658 → 0.904406 → 0.855654 ms，
收益主要来自 tail 和归并头缓存，合并加载几乎无影响。

## 5. 正确性与编译资源

`validate_q3_dpas.py` 默认运行 **18 个回归用例**，两路径均与 NumPy FP32 reference 比较，
并断言优化前后逐元素相同。最坏最大绝对误差 **0.000386536**，阈值仍为 **0.02**，未放宽。

覆盖：部分 query tile、混合 past/query 长度、跨页 tail、乱序物理 pages、独立稀疏选择、
连续/有间隙/跨页的选择模式、空列、tail-only、完全空 attention、top-k=0/1、
page=8/16/32、gate 开关、带 offset/padding 的 query、W=1/4/8、HT=1/2/4/8、r=2/4/8/16。
输出预填 NaN 检查漏写，尾部 sentinel 检查末端越界写；这不等同于完整内存 sanitizer。

另外通过了原测试驱动的 decode-shaped 输入检查（batch=4，past=1/15/16/17/2048，top-k=4）；
并未改变生产 decode 应使用 scalar kernel 的策略。

默认配置 IGC 汇编：两路径均为 `numGRF=256`，未发现 GRF spill/scratch 访存；
旧路径有 flag spill store/load 注记，优化路径没有该注记。
远程审计产物位于 `/tmp/qsa_dpas_opt_20260910/`。

扩展测试发现的**原始基线限制**，未纳入通过集合，亦未在本次优化中修复：
- `Dh=128, Dv=64, W=2, HT=4`：当前远程 IGC 报 `Internal Compiler Error: Floating point exception`。
- `HT=16, TT=1`：旧 `load_vnni_cols<1>` 的 transpose 目标大小不符合当前 CM frontend 要求。
这两项均不影响模型的 `Dh=Dv=256, W=4, HT=4` 配置。

## 6. 复现入口

以下参数传给远程 `.venv/bin/python`，工作目录为 `/mnt/river/qsa/cm_kernel`，
并设置 `CM_FE_DIR=/mnt/river OPENBLAS_NUM_THREADS=1`；建议外层使用 `timeout 300`：

- 回归：`validate_q3_dpas.py`。
- 默认 A/B：`validate_q3_dpas.py --benchmark --sizes 1024 2048 4096 8192`。
- 消融：`validate_q3_dpas.py --benchmark --ablation --sizes 1024 2048 4096`。
- 稀疏 A/B：`validate_q3_dpas.py --benchmark --block-topk 4 --sizes 1024 4096`。
- 原有测试：`test_q3_sparse_attention_dpas.py --sizes 17 64 257 --check --iters 3`。
- 原始路径：给原有测试增加 `--baseline`；正确性失败现在会断言并返回非零状态，而不只是打印 FAIL。

本地 `cm_kernel/.env` 是 `CM_FE_DIR` 的占位模板；脚本不自动加载它。
远程测试一直显式设置真实路径，未用占位值覆盖远程环境。