# 长上下文 Q2 top-k：精确 SIMD 阈值选择实测

日期：2026-09-10。范围仅 `cm_kernel/`，不修改插件、不增加自动 dispatcher。
**更新：第9节已完成真实完整 prefill8K/16K/32K/64K 与 B1 decode16K/32K/64K
同输入 A/B/C；前8节保留为先前阶段证据。最终推荐显式 `fast` / `fast-wg16`，默认不变。**
所有 GPU 执行在 `openvino-ci-74@10.239.140.245` 的 Arc B580；串行、独占
`.qsa_gpu.lock`，`CM_FE_DIR=/mnt/river OPENBLAS_NUM_THREADS=1`。
未采用任何 Explore 报告的算法结论、计数或性能数字。

## 1. 交付与显式接口

新增：
- `include/qsa_topk_fast.hpp`：SIMD64 有序键加载、整数前缀和。
- `kernels/qsa_topk_radix_fast.cm`：一个 work item / row，256-key 精确 cohort 缓存。
- `kernels/qsa_topk_radix_wg.cm`：一个 WG / row，WG8/16，SIMD64 全扫描，无候选容量限制。
- `validate_topk_fast.py`：同一 device score buffer 的 scalar / cooperative / fast / fast-wg
  精确对比、安全回归、实际 Q0/Q1/DPAS 行采样、交替微基准。
- 本文及 `topk_long_logs/` 原始证据。

修改：
- `qsa_score_chunked.py`：新增 keyword-only `finalizer=None`。
  `None` 完全保留旧 `cooperative` 行为；可显式选 `scalar`、`cooperative`、`fast`、`fast-wg`。
  `fast` 用 LWS1；`fast-wg` 用调用者的 `wg=8/16`，不根据 shape 自动选择。
- `test_qsa_pipeline.py`：`--topk-finalizer` 显式 override，并按实际内核记录事件名称。
- `benchmark_long_context.py`：同名显式参数；省略参数保持旧行为。

**旧 `qsa_topk_finalization.cm`、`qsa_topk_cooperative.cm`、`qsa_common.hpp` 均未修改。**
新内核的输入/输出顺序和 `QSA_SCORE_ROW_CHUNK` ABI 与旧 finalizer 一致。
global token 用于 metadata/selected/count；chunk-local row 用于 scores。
不新增 GPU 全局 scratch，不改变原 128MiB score scratch 预算。

## 2. 算法与正确性

### 有序整数键与阈值

沿用原始位变换：负 float 位模式 XOR `0xffffffff`，非负 XOR `0x80000000`。
没有浮点加减、量化、fast-math、直方图动态 GRF 更新。
因此 `-0` 与 `+0` **按原 bit-key 语义区分**，不是 NumPy 普通 float 排序的零相等语义。
测试 oracle 对有限输入按 `(ordered_key 降序, block_id 升序)` 排序，最终输出 ID 升序。

维护包含第 k 大键的闭区间 `[lo, hi]`，取整数上中点 `mid`：

1. SIMD64 分块统计 `rank = count(key >= mid)`，同时求这一集合的实际最小键，及
   `key < mid` 集合的实际最大键。尾块用 masked gather，不读取 padding/下一行。
2. 若 `rank >= k`，令 `lo = min(key >= mid)`；否则令 `hi = max(key < mid)`。
3. `rank == k` 时实际最小上半键就是精确阈值，可停止；否则直到 `lo == hi`。

每一步至少将整数区间减半，所以至多 32 轮；利用实际 min/max 跳过的是空键区间，
不是基于熵、分布或容差的近似停止。全相同键无需逐 bit 分辨，仍是严格结果。

### 单 work-item cohort 缓存及回退证明

`above` 精确表示 `key > hi` 的个数，`active` 精确表示 `[lo, hi]` 内的个数。
当选择上半区时 `active = rank - above`；选择下半区时
`active = active + above - rank; above = rank`。

只有 `active <= CAP` 才正常尝试缓存（默认 CAP256）。再完整扫描一遍，用匹配 lane 的
bit mask + find-first-bit 将 **全部** 区间键复制到寄存器；仅复制匹配项，不对全行做 scalar histogram。
写入前检查 `filled < CAP`，并在末尾检查 `!overflow && filled == active`。
只有两项均满足才启用缓存；缓存中的 rank 加上固定的 `cache_above` 即完整行 rank。
之后的区间是缓存区间的子集，因此缓存外已确定的上下部分不会改变选择结果。

**容量不足不会截断选择集合。** 溢出/计数不匹配时保留原区间、永久使用完整行扫描。
`QSA_TOPK_CAPACITY=0` 保留无缓存初版；验证专用 `QSA_TOPK_FORCE_COMPACT=1` 可在
最初大区间强制触发真实 overflow 分支，已用 CAP64/256 对长行验证。
超容量 ties 也不会被丢弃；可能不启用缓存，但仍能精确停止。

### 输出顺序与 WG 协作

确定阈值后再完整统计 `greater`，取 `k-greater` 个最早出现的阈值相等项。
SIMD inclusive prefix 为稳定的 ascending emission 计算位置，masked scatter 只写被选 lane。
没有缓存 accepted ID 的额外上限；输出总量严格为 k。

WG 版不使用 cohort 缓存：每 worker 负责连续、64 对齐的候选范围，空 worker 贡献中性值。
每轮由 leader 汇总 rank/min/max，通过独立 SLM state 广播新边界。
WG16 使用 `(16+1)*16 = 272` 字节 SLM；WG8 使用 144 字节。
每个 barrier 前都有 local fence；没有 peer global-memory 回读。
输出前 leader 按连续 worker 顺序分配 offset 和 tie quota，worker 内再做稳定 SIMD emission。

### 输入与空路径约定

- 稀疏输入要求 finite；测试显式拒绝 NaN。GPU 内核不额外插入 NaN 检测/报错 dispatch。
- K0、n0、dense `k==n` 不加载 score；dense 即使 producer 留下 NaN 也输出完整升序 ID。
  新内核即使 `QSA_DENSE_BYPASS=0` 也无需读取 dense 分数，输出语义不变。
- 先检查 global/chunk 行边界；未使用 selected 槽写 `-1`。
- masked 最后一块与 global row 输出保护通过非 16 对齐 stride、partial row chunk 和 guards 测试。

## 3. 正确性验证结果

最终 `topk_fast_final.log`：

|验证|结果|
|---|---|
|finalizer 参数组合|52 组通过；每组包含四个路径精确对比及分块重放|
|显式 fast/fast-wg ChunkedQ2 helper|144 组通过，每组两次重放|
|原默认分块回归 `validate_q2_chunked.py`|148 组通过，原脚本未改|
|`fast` 完整 Q0→Q3、cap17、full-ref|9 个集成调用通过|
|`fast-wg` 完整 Q0→Q3、cap17、full-ref|9 个集成调用通过，prefill WG8 / decode WG16|
|真实 64K-history B1 decode 完整流水线|通过，包括真实 history、own-score IDs、Q3、guards、全 payload replay|

52 组覆盖：K0/1/512；n513/1024/4096/8192/16384/32768/32769；旧 ABI 与 chunk ABI；
混合序列、空序列；dense poison；全相同/全零/负数/正负零/相邻 float/精确 ties；
capacity0/64/256/512 和强制 overflow；WG8/16；chunk3/5/17、输出区间外保持不变、score 不被改写。
144 组覆盖两类 scorer（partition/DPAS）、bypass on/off、row_cap1/3/17、随机/零 summary、重放；
此外验证空 batch 不 dispatch。

实际 Q0/Q1/DPAS 分数，**非随机 indexer summary 替代品**：

|prefill tokens|采样 global row begin|采样行数|最大完整候选数|score SHA256|
|---:|---:|---:|---:|---|
|4096|4080|16|1024|`3ba95f920491acccdc61a44f3a9329a97cb8404285b70d8e12e9a34c7c475e34`|
|16384|16371|13|4096|`a1b7228bb118fcda934d380538831cb25f76e82307876960fe6746c01ec8194b`|
|65536|65535|1|16384|`af92984162c981752b53d639d323ab063e8a2969d91b9fd38fa87f4736142aad`|

这里长 prefill 只执行 Q0/Q1 和最后 score chunk，不执行完整 prefill Q2/Q3。
真实 64K decode 的 history 另由 `benchmark_long_context.py` 全量流式验证。

## 4. 正式 finalizer-only 微基准

每个 shape/cache/variant **100 warmup + 20 samples**，variant 次序正反交替。
scalar/cooperative/fast/fast-wg 读取 **同一个 device score buffer**，各有独立输出；
计时前后都检查 exact ID/count/guard，记录 OpenCL event 时间，不含编译、拷贝、参考或 flush。
`cold96MiB` 在每个被测 dispatch 之前独立 flush；no-flush 不宣称数据必然驻留 cache。
所有下表为 mean ms，median/p95/全部样本在原始 JSON。

“候选区间”是一个连续 causal prefill slice 内的实际 n 范围，不是所有行固定 n。
例如 rows4096、max4096 时实际 n=3072..4096；没有使用错误 dense 计数。
下面所有行 K512，均为 sparse，dense 数为 0。

### 4.1 独立 uniform[0,10) score rows（最终复跑）

来源：`topk_fast_final.log`。old WG 与 fast-wg 均使用 **WG16**。

|rows|候选区间|cache|scalar|old WG16|fast CAP256|fast WG16|
|---:|---|---|---:|---:|---:|---:|
|1|4096|no-flush|1.972838|0.147026|0.138890|0.022275|
|1|4096|cold96MiB|1.987979|0.149114|0.153052|0.024427|
|256|4032..4096|no-flush|2.007125|1.004583|0.145322|0.164869|
|256|4032..4096|cold96MiB|2.014760|1.011849|0.158953|0.170322|
|4096|3072..4096|no-flush|12.663671|13.484312|1.281536|2.479005|
|4096|3072..4096|cold96MiB|12.698807|13.482390|1.292239|2.479536|
|1|16384|no-flush|7.829760|0.494672|0.596708|0.062833|
|1|16384|cold96MiB|7.887885|0.499312|0.650968|0.067510|
|256|16320..16384|no-flush|8.190630|3.456875|0.887718|0.495130|
|256|16320..16384|cold96MiB|8.276182|3.476276|0.955302|0.516885|
|4096|15360..16384|no-flush|56.350682|49.699802|10.197260|7.508135|
|4096|15360..16384|cold96MiB|56.354765|49.704687|10.225973|7.506562|

首轮正式 100/20 另保留于 `topk_fast_benchmark.log`，主要优劣方向相同。
但 rows256/max16384 的 fast CAP256 首轮 no-flush 1.251114 ms、复跑 0.887718 ms，
存在明显绝对时间波动；未采集功率/频率，不将其归因于 DVFS，也不只引用最短样本。

### 4.2 真实长 prefill 最后一行 score 重放

来源：`topk_fast_actual_benchmark.log`。每个 n 从真实 Q0/Q1/DPAS 取得最后一行；
多行 case 复制这一行，然后用各行自身 causal n 截取。
**这些是高度相关的行重放，不是完整 prefill，也不是独立 query 的实测吞吐。**

|rows|候选区间|cache|scalar|old WG16|fast CAP256|fast WG16|
|---:|---|---|---:|---:|---:|---:|
|1|4096|no-flush|1.973265|0.141541|0.105276|0.016796|
|1|4096|cold96MiB|1.989130|0.143364|0.119547|0.018603|
|256|4032..4096|no-flush|2.001479|0.957067|0.109057|0.123442|
|256|4032..4096|cold96MiB|2.012635|0.963140|0.123588|0.129901|
|4096|3072..4096|no-flush|12.647869|13.090400|0.996401|2.045765|
|4096|3072..4096|cold96MiB|12.675499|13.086682|1.006234|2.043828|
|1|16384|no-flush|7.810380|0.483161|0.453343|0.051697|
|1|16384|cold96MiB|7.867416|0.486661|0.506406|0.055838|
|256|16320..16384|no-flush|8.362432|3.393422|0.946797|0.410380|
|256|16320..16384|cold96MiB|8.407895|3.393171|1.002125|0.441275|
|4096|15360..16384|no-flush|56.405875|49.209916|7.216770|6.032937|
|4096|15360..16384|cold96MiB|56.409520|49.202854|7.244401|6.025666|

## 5. 真实完整 64K-history decode（额外集成证据）

来源：`topk_decode65536_fast_wg.log`。B1、past65536、q1，K512，n16384；
1 个 sparse row、0 个 dense row，实际 attention positions=2049。
Q2 scorer partition16，finalizer 显式 fast-wg/WG16，其余保持现有 optimized pipeline。
每个 cache100/20，score scratch65792 B，split scratch396288 B。

|cache|Q2 finalizer mean ms|完整 GPU sum mean ms|host elapsed mean ms|
|---|---:|---:|---:|
|cold96MiB|0.053578|0.160711|0.210856|
|no-flush|0.064541|0.168794|0.220216|

该完整流水线只测新路径，没有本轮交替的完整旧 pipeline，因此**不据此声称完整 pipeline A/B 加速比**。
与旧约0.49ms finalizer 的严格同分数对照见第4节，而非跨运行推断。
原历史64K prefill Q2topk约537.9ms不属于本次测量，不能乘微基准比例推算新完整 prefill 时间。

## 6. 原型状态、建议与限制

|方案|状态与依据|
|---|---|
|无缓存单 work-item SIMD threshold|保留 CAP0 编译选项和回归；首轮3/3短测 rows256/n16384约1.218ms，B1约0.771ms，比 old WG16 B1慢；不推荐用于长 B1|
|CAP256 单 work-item|保留并提供 `finalizer='fast'`；多行 max4096 比 scalar 明显快；不推荐覆盖所有 decode|
|WG SIMD threshold|保留并提供 `finalizer='fast-wg'`；正式 B1/max16384 uniform no-flush 比 old WG16快约7.87倍；真实行重放约9.35倍|
|原 scalar/old cooperative|全部保留、仍是默认；没有因新方案出现而删除 fallback|

初版、缓存版、WG版三个短测日志明确为 **3 warmup/3 samples**，不能替代正式结果。
缓存版曾出现一次编译错误：`half` 是 CM 类型名，不能作局部变量；改名 `part` 后编译通过。
未发生为了通过精度测试而放宽容差、限制候选数或丢弃 ties。

当时的下一步建议（历史记录；完整集成测量现已见第9节，仍不提供自动策略）：

- B1 的 n4096/16384：优先验证 fast-wg WG16；已有随机与真实行证据。
- 多行 n约3K..4K：fast CAP256 更合适；rows4096 uniform no-flush 相对 scalar约9.88倍。
- 多行 n约15K..16K：fast-wg WG16 更合适；rows4096 uniform 相对 scalar约7.51倍。
- 不能从两个端点拟合安全 crossover，也没有 B4、其他 K/batch/设备的性能分派证据。
- **集成 prefill 的现有 `wg` 是8**；本文微基准的 WG性能表是16。
  `--topk-finalizer fast-wg` 不偷偷改这个既有参数；不要把该 CLI 的 prefill WG8 当成表中 WG16。
- 当时尚未测完整16K/64K prefill；现已由第9节的完整流水线 A/B/C 补齐。
- 未采集硬件带宽/occupancy/GRF spill 遥测；不声称达到某一 roofline 百分比。
- finite之外、模型质量、插件集成、跨平台性能不在本次结论内。

## 7. 复现命令与证据审计

所有以下命令在本地通过前缀执行：
`ssh -o BatchMode=yes openvino-ci-74@10.239.140.245 'cd /mnt/river/qsa/cm_kernel && ...'`。
省略号替换为以下完整远端命令；不要并行启动 GPU 作业。

1. 最终验证 + uniform 正式基准：
   `CM_FE_DIR=/mnt/river OPENBLAS_NUM_THREADS=1 timeout 300 flock -n /mnt/river/qsa/cm_kernel/.qsa_gpu.lock .venv/bin/python validate_topk_fast.py --benchmark --warmup 100 --samples 20`
2. 真实长 score 行重放正式基准：
   `CM_FE_DIR=/mnt/river OPENBLAS_NUM_THREADS=1 timeout 300 flock -n /mnt/river/qsa/cm_kernel/.qsa_gpu.lock .venv/bin/python validate_topk_fast.py --benchmark-only --distribution pipeline-row --warmup 100 --samples 20`
3. 原默认148组：
   `CM_FE_DIR=/mnt/river OPENBLAS_NUM_THREADS=1 timeout 300 flock -n /mnt/river/qsa/cm_kernel/.qsa_gpu.lock .venv/bin/python validate_q2_chunked.py`
4. 单线程新路径集成：
   `CM_FE_DIR=/mnt/river OPENBLAS_NUM_THREADS=1 timeout 300 flock -n /mnt/river/qsa/cm_kernel/.qsa_gpu.lock .venv/bin/python test_qsa_pipeline.py --optimized --topk-finalizer fast --score-row-cap 17 --full-ref`
5. WG新路径集成：
   `CM_FE_DIR=/mnt/river OPENBLAS_NUM_THREADS=1 timeout 300 flock -n /mnt/river/qsa/cm_kernel/.qsa_gpu.lock .venv/bin/python test_qsa_pipeline.py --optimized --topk-finalizer fast-wg --score-row-cap 17 --full-ref`
6. 真实长 decode：
   `CM_FE_DIR=/mnt/river OPENBLAS_NUM_THREADS=1 timeout 300 flock -n /mnt/river/qsa/cm_kernel/.qsa_gpu.lock .venv/bin/python benchmark_long_context.py --phase decode --size 65536 --topk-finalizer fast-wg --warmup 100 --samples 20`

`topk_timing_audit.log` 已从最终 uniform + actual 两份日志重算 **24行、1920个计时事件**
的 mean/median/p95（容差1e-12ms），并验证 ENV 内全部源码哈希等于当前远端文件。
还核对64K decode的两组20×7事件及GPU总和。
首轮正式与短测也保留，不混入最终表的样本数。

编辑器对四个修改/新增 Python 源报告无错误；Pylance语法检查通过。
签名兼容分析对原验证器两处动态 `**opts` 调用报不兼容，但这些实际调用已经由未修改的
默认148组完整运行覆盖并通过；没有为静态推断而改动旧测试。

## 8. 完整交付文件清单

源码/文档共8个（相对 `cm_kernel/`）：
`include/qsa_topk_fast.hpp`、`kernels/qsa_topk_radix_fast.cm`、
`kernels/qsa_topk_radix_wg.cm`、`validate_topk_fast.py`、`qsa_score_chunked.py`、
`test_qsa_pipeline.py`、`benchmark_long_context.py`、`TOPK_LONG_OPTIMIZATION_CN.md`。

证据共13个（均在 `topk_long_logs/`）：
- `topk_fast_preliminary.log`
- `topk_fast_validation_initial.log`
- `topk_fast_compact_preliminary.log`
- `topk_fast_wg_preliminary.log`
- `topk_fast_validation.log`
- `topk_fast_benchmark.log`
- `topk_default_chunk148.log`
- `topk_pipeline_fast.log`
- `topk_pipeline_fast_wg.log`
- `topk_decode65536_fast_wg.log`
- `topk_fast_actual_benchmark.log`
- `topk_fast_final.log`
- `topk_timing_audit.log`

## 9. 完整 QSA slice 同输入 A/B/C（最终协议 v2）

### 9.1 范围、共享状态与测量方法

新增 `benchmark_topk_pipeline.py`。配置不变：D2560、Hq24/Hkv2、Dh/Dv256、
Hidx4/Di128/rot32、r4/PA16、K512、gated causal、B1。
prefill 是 past0 的全部8192/16384/32768/65536个 query；decode 是真实 past16384/32768/65536 后追加1个 query。
不是 finalizer-only、复制同一 score 行、最后一块或模型/plugin 测量。

三个 view **共享一个 Fixture、State、scorer 对象、tile map、全部输入、score scratch、
selected/count/output**；只替换 finalizer program、对应 WG 和事件标签。
Q0/Q1/Q3 及 scorer 的程序对象与 dispatch 均相同；Q1 都是 DPAS，所以不存在旧 scalar-Q1
对比新 DPAS-Q1 的 near-tie 选择差异。固定 past/输入下 Q0/Q1 的 cache stores 可确定性重放，
所有 view 只能串行使用，不能并发。没有保留三个64K大 Fixture。

- `legacy` = **当前 optimized pipeline 的旧 finalizer**，不是整个旧 baseline pipeline：
   这四种 sparse prefill 用 scalar，decode 用 old cooperative WG16。
- 对比 `fast` CAP256 与 `fast-wg16`；每个完整 slice 包含 Q0、Q1-project/prepare、
   每块 score→topk、Q3（decode 包含 split-P16 + reduce）。不替换 Q3 来夸大 Q2收益。
- 每个 shape/cache/variant **100 warmup + 20 samples**，逐轮正序/逆序交替；
   cold96MiB 在**每次 warmup 和 sample 的整个 slice 前** flush，flush自身不计时。
   no-flush不保证数据驻留。8K prefill与全部decode测两种cache，16/32/64K prefill测cold主协议。
- JIT、输入/历史、分配、校验、输出读取均在计时外；**20轮样本之间也不做readback/日志输出**。
   GPU数值是所有event之和；host是enqueue→finish，不等于首末GPU时间戳跨度。
   `SAMPLE`按实际执行顺序于测量后统一打印，`BENCH`保存各event、mean/median/线性p95及host原始样本。
- 远端唯一 `.qsa_gpu.lock`，无重叠GPU工作；每个形状 `timeout 900`，同步串行执行。

**协议修正记录，不删异常样本：** 首轮在第20轮每个variant后读取完整payload，导致接下来的
variant因长CPU停顿出现最后样本尖峰，例如64K legacy909.350ms。首轮完整保留在
`topk_long_logs/initial_ab_readback/`，不纳入最终表。修正为整轮完成后读回，并使cold warmup
也使用flush，重新执行全部7形状/11个cache组合。本节只引用v2完整复跑，没有挑选最快样本。

### 9.2 最终 cold96MiB 全部三路径（ms，算术mean）

|phase/size|variant|完整GPU|Q2topk|Q2score|host|GPU median|GPU p95|
|---|---|---:|---:|---:|---:|---:|---:|
|prefill/8192|legacy|32.466102|6.750723|0.241703|32.500743|32.464164|32.531836|
|prefill/8192|fast|26.246716|0.518510|0.241468|26.281864|26.237914|26.303680|
|prefill/8192|fast-wg16|27.729123|2.043666|0.241052|27.763875|27.729842|27.758295|
|prefill/16384|legacy|86.889569|29.298848|0.982691|86.937005|86.864579|87.050194|
|prefill/16384|fast|59.785345|2.246827|0.981608|59.835308|59.780933|59.958642|
|prefill/16384|fast-wg16|63.393382|5.879015|0.982025|63.442171|63.343226|63.630902|
|prefill/32768|legacy|248.546678|120.207496|4.170100|248.631828|248.577439|248.721143|
|prefill/32768|fast|139.129548|10.761059|4.169960|139.228769|139.081606|139.376962|
|prefill/32768|fast-wg16|145.799377|17.433632|4.168163|145.897117|145.801085|145.946411|
|prefill/65536|legacy|839.001299|537.895158|19.999742|839.257909|839.015126|839.214976|
|prefill/65536|fast|365.636074|64.428110|20.015027|365.946594|365.598354|365.922055|
|prefill/65536|fast-wg16|358.252814|57.097653|19.987283|358.562107|358.229346|358.543805|
|decode/16384|legacy|0.228736|0.143234|0.011749|0.279949|0.228642|0.229273|
|decode/16384|fast|0.197809|0.112239|0.011791|0.248142|0.197601|0.198439|
|decode/16384|fast-wg16|0.105866|0.020348|0.011734|0.155259|0.105986|0.106564|
|decode/32768|legacy|0.428919|0.325088|0.014802|0.478837|0.429059|0.429597|
|decode/32768|fast|0.371101|0.267208|0.014859|0.421199|0.371090|0.371684|
|decode/32768|fast-wg16|0.131742|0.027822|0.014786|0.180447|0.131819|0.132618|
|decode/65536|legacy|0.618861|0.514255|0.025130|0.667944|0.602758|0.710424|
|decode/65536|fast|0.584903|0.478140|0.025260|0.633725|0.563433|0.665930|
|decode/65536|fast-wg16|0.156919|0.051849|0.025255|0.207503|0.152497|0.175441|

64K decode cold仍有明显样本波动，保留完整20个样本；不根据缺失的频率/功耗遥测归因。
CPU readback artifact已从正式协议排除，不表示其它运行波动也被解释。

### 9.3 推荐路径与相对 legacy 实际收益

|phase/size|显式选择|完整GPU old→new ms|完整GPU加速|Q2topk old→new ms|topk加速|
|---|---|---:|---:|---:|---:|
|prefill/8192|fast|32.466102→26.246716|1.237×|6.750723→0.518510|13.019×|
|prefill/16384|fast|86.889569→59.785345|1.453×|29.298848→2.246827|13.040×|
|prefill/32768|fast|248.546678→139.129548|1.786×|120.207496→10.761059|11.171×|
|prefill/65536|fast-wg16|839.001299→358.252814|2.342×|537.895158→57.097653|9.421×|
|decode/16384|fast-wg16|0.228736→0.105866|2.161×|0.143234→0.020348|7.039×|
|decode/32768|fast-wg16|0.428919→0.131742|3.256×|0.325088→0.027822|11.684×|
|decode/65536|fast-wg16|0.618861→0.156919|3.944×|0.514255→0.051849|9.918×|

未观察到新方案相对legacy的完整GPU mean回退，但**统一选择WG16并非最优**：
8/16/32K prefill的WG16比fast完整GPU分别慢约1.482/3.608/6.670ms；64K反过来fast比WG16慢7.383ms。
这些是实测离散点，不构造32K到64K之间的交叉阈值，不外推到>64K、B4、其它K/设备或模型数据分布。
显式选择整个调用的finalizer，不按chunk另行推测派发；所有chunk都保留完整候选列与global metadata。

### 9.4 no-flush 对照（所有路径完整GPU / Q2topk，ms mean）

|phase/size|legacy GPU/topk|fast GPU/topk|fast-wg16 GPU/topk|
|---|---:|---:|---:|
|prefill/8192|32.450862 / 6.752906|26.217878 / 0.518494|27.726992 / 2.043416|
|decode/16384|0.212852 / 0.143182|0.181696 / 0.111995|0.089946 / 0.020161|
|decode/32768|0.334076 / 0.263416|0.287008 / 0.216364|0.093075 / 0.022385|
|decode/65536|0.572002 / 0.491526|0.533169 / 0.452739|0.129352 / 0.048942|

### 9.5 正确性与内存

1. baseline完整Q0→Q3经 `benchmark_long_context.validate` 检查所有Q0/Q1投影/缓存、
    **全部query own-score精确选集**、counts/padding、有限输出、guards与完整重放。
2. 对每个chunk只运行一次score，三个finalizer读取同一个device scores；每次比较**全体行**
    selected/count与baseline（包括当前chunk以外不应改动的行），并验证score原始bit SHA256不变。
3. 每个variant计时前、每个cache测量后重新运行带额外guard WG的完整slice，检查所有payload
    原始bit SHA256与baseline一致，包括output、iq、projected、缓存、最终score scratch、split scratch。
    每个输出元素都检查finite；最后实际timed共享payload也在下一次写之前检查。
    共享buffer不会保留其它variant最后一帧，因此其它variant的post-check是显式完整重放，
    **不声称保存或检查了每个timed sample的输出**。
4. 每个variant前后均以**自身真实选集**做采样NumPy Q3，atol=rtol=0.002不变。
    prefill8/16/32/64K Q3分别41/45/57/104行（含chunk与dense边界），最大误差0.000241339207；
    NumPy Q2分别29/37/51/98行，最大1.907348633e-6。decode全query，Q3最大1.267716289e-5。
    所有variant selected/count/output bit完全一致，零选择差异。
5. decode历史为真实外部x/K/V经optimized Q0+DPAS Q1产生，全部历史Q1流式NumPy检查，
    不使用随机summary冒充历史。历史生成/校验不在timed slice。
6. score上限仍为128MiB（含256B guard）。64K为2047×16384、33chunks、134152448B；
    整个GPU slice70events。最终64K prefill峰值host RSS10661352KiB（约10.17GiB），
    是包含CPU参考/读取的进程高水位，不是GPU显存遥测。未测量峰值GPU resident memory。

最终额外回归：52 finalizer参数组、**288 helper组（含固定WG8/WG16新别名，每组两次重放）**、
未改的148 chunk组，以及 `fast-wg16` cap17 full-ref全部9个集成调用通过。
CPU dispatch与raw-log审计6测试通过（含5种损坏证据拒绝）。无CM内核修改，没有放宽任何GPU正确性容差。

### 9.6 可用API与默认策略

- `ChunkedQ2(..., finalizer=None)`及新增`finalizer='legacy'`都保留调用者原cooperative/wg策略。
- `finalizer='fast'`选CAP256/LWS1；`'fast-wg'`仍继承调用者wg（集成prefill默认8）。
- 新增`'fast-wg8'`、`'fast-wg16'`明确固定工作组，解决旧CLI无法直接选prefill WG16的问题。
- `test_qsa_pipeline.py --topk-finalizer ...`与`benchmark_long_context.py --topk-finalizer ...`
   接受相同选项；不传或传legacy默认完全不变。短/未测形状继续使用legacy，**没有auto-fast**，
   避免用离散点虚构性能阈值。推荐flag只针对9.3表中的实测主配置。
- finite输入、ordered-key/tie/ascending语义及CAP不足完整扫描fallback均保持第2节契约。

### 9.7 复现与原始证据

远端命令均通过 `ssh -o BatchMode=yes openvino-ci-74@10.239.140.245`，工作目录
`/mnt/river/qsa/cm_kernel`，串行执行；命令前缀为
`CM_FE_DIR=/mnt/river OPENBLAS_NUM_THREADS=1 timeout 900 flock -n /mnt/river/qsa/cm_kernel/.qsa_gpu.lock .venv/bin/python`。

- 新完整A/B：前缀后接 `benchmark_topk_pipeline.py --phase prefill --size 65536 --warmup 100 --samples 20`。
   支持size8192/16384/32768/65536；decode用 `--phase decode --size 65536 --cache cold96MiB no-flush`。
   默认variants为legacy/fast/fast-wg16，默认cache为cold96MiB。
- 推荐单路径：`benchmark_long_context.py --phase prefill --size 32768 --topk-finalizer fast`；
   64K prefill或长decode改用 `--topk-finalizer fast-wg16`。
- 集成API验证：`test_qsa_pipeline.py --optimized --topk-finalizer fast-wg16 --score-row-cap 17 --full-ref`。
- CPU审计：`python3 analyze_topk_pipeline.py --check-sources`，重算**33 BENCH、660 slices、9360events**，
   核对SAMPLE实际正逆序、所有stage/total/host mean/median/p95、全payload hashes、history、guards、source hashes。
   `python3 -m unittest test_topk_finalizer_options test_analyze_topk_pipeline -v`。

最终证据：`topk_long_logs/pipeline_ab_prefill{8192,16384,32768,65536}.log`、
`pipeline_ab_decode{16384,32768,65536}.log`、`pipeline_timing_audit.log`、
`pipeline_dispatch_validation.log`、`pipeline_chunk148.log`、`pipeline_wg16_regression.log`。
审计日志列出全部7份最终raw log SHA256；ENV保存测量时源码hash，不用后续源码冒充测量版本。

原 `LONG_CONTEXT_PERFORMANCE_CN.md`、`LONG_CONTEXT_ROOFLINE_CN.md`、原14份logs、
SHA256SUMS及分析脚本/测试均**未修改**。原分析器会严格拒绝当前已演进的3个Python源码hash；
新增 `audit_long_context_history.py` 用可验证reverse delta恢复这3个文件到临时source-root，
每一字节必须匹配历史ENV hash，其他53个源码同样验证后才运行原分析器；不跳过source检查。
`python3 audit_long_context_history.py`验证原报告逐字相同，`--test`在该原始source-root上运行
未修改的12个历史测试。新性能只链接至本文，不重写旧数字/roofline模型。

