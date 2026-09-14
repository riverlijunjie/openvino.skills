# Q2 SCORE SCRATCH：按 query rows 有界复用

## 范围与不变量

仅修改 `cm_kernel/` standalone；没有 model、OpenVINO plugin、Q0/Q1/Q3 kernel
或已有 Python helper 默认值的改动。日期：2026-09-10。

- **不是 candidate columns 分块**：每个 query 始终对完整可见历史评分，原有
  partition scorer 的列并行分区仍在同一次评分 dispatch 内完成。
- 执行顺序：`row chunk → score full history → finalize → next row chunk`。
  一次分配、始终复用同一个 device scratch。finalize 后不保留该 chunk 的 scores。
- 完整 `iq/query`、metadata、`selected/counts` 是全局 persistent buffers；Q3 在所有
  chunk finalize 完成后消费完整选择结果。没有局部 top-k 列表，也没有全局 merge。
- clops 单一 **in-order queue**；计时内不分配、不复制、不重建 map、不清零 scratch、
  不做 chunk 间 finish。正确性验证的读回与同步全部在计时外。
- score dot、per-head ReLU、head reduction、scale 和 finalizer radix/tie 顺序未改。
  exact 对比限定为 **相同 scorer / 相同输入 / 相同 bypass 设置**，不声称 partition
  与 DPAS 或 Q1 scalar 与 DPAS 的 FP32/FP16 运算彼此 bit-exact。

## 文件与 API

新增 `qsa_score_chunked.py`：

- `plan_score_chunks(subsequence_begins, max_blocks, *, budget_bytes=128MiB,
  row_cap=None, guard_elements=0, unchunked=False)`：纯 Python，不 import NumPy/clops。
  `max_blocks` 为完整历史的 padded stride，至少 8 且是 8 的倍数。
  返回 immutable `ScorePlan` / `ScoreChunk`：row_begin/end、tile_offset/count、
  concatenated tile map、scratch rows/payload/allocation bytes。
- `ChunkedQ2(cfg, batch, max_blocks, *, backend='dpas', cooperative=False, wg=16,
  partition_blocks=16, dense_bypass=True, budget_bytes=128MiB, row_cap=None,
  guard_elements=64, unchunked=False)`：拥有两个 kernel、一个 map、一个 scratch。
- `enqueue(inputs, selected, counts, *, guard_groups=0)`：inputs 顺序为
  iq、summary、past、subsequence_begins、pages、page_begins。输出保持全局 token 索引。
- `score(chunk, inputs)` / `finalize(chunk, past, begins, selected, counts)`
  用于逐块验证；`read()` 只读固定 scratch 并检查 guard，不能获取完整历史 score matrix。
- budget 必须在 4 bytes..128MiB；含 guard 向下按**完整 row**取整。
  budget 连一整行都放不下时抛 ValueError，不增加 budget、不截断 candidate width。
  NT=0 无 chunk/无 dispatch；零长 map、空 score 使用非零 dummy，避免 clops 零字节 copy。
  `unchunked=True` 仅在完整 scores（含 guards）不超过 budget 时允许。

修改四个 kernel，`QSA_SCORE_ROW_CHUNK` **默认 0** 保持旧 ABI：

| Kernel | 宏为 1 时追加在旧参数末尾 | 索引 |
|---|---|---|
| `qsa_score_partition.cm` | `uint32 row_begin, row_end` | global token = local global_id + row_begin；scratch row = token - row_begin |
| `qsa_score_tile_dpas.cm` | `uint32 row_begin, row_end, tile_offset` | map 仍存 global token；scratch row = token0 + q - row_begin |
| `qsa_topk_finalization.cm` | `uint32 row_begin, row_end` | global metadata/output，local score row |
| `qsa_topk_cooperative.cm` | `uint32 row_begin, row_end` | global group token + row_begin，local score row；guard 在 barrier 前 |

DPAS 每个 chunk 的 map 在 host 构造时按 sequence 切开，不能跨 sequence；query
descriptor 的 valid rows 同时限制到 sequence end、row_end、16。即使 row_cap=1/3/17
也不加载下一块 query，更不会写下一块 score rows。8-wide store 使用完整 padded stride。
所有 scratch byte offsets 保持远小于 uint32 极限，不再出现大矩阵的 4GiB 地址风险。

`test_qsa_pipeline.py` 的 baseline 和 optimized 都默认使用 bounded owner。
原 `TopKFinalizer`、`PartitionScorer` 未修改。`Pipeline(..., score_options=...)`
可显式覆盖 owner 选项；CLI 传入 `SCORE_OPTIONS`。真实 decode 历史仍由 scalar Q0/Q1
生成并验证，历史初始化只需 Q0/Q1，给其未使用的 Q2 scratch 强制 1 row，避免隐藏大矩阵。

## 内存：1024 → 65536 prefill（理论值，不是实测性能）

单序列 past0，r=4，`MB=ceil8(N/4)`，FP32。
旧 payload = `N * MB * 4`；新 guard-free 理论 row 数为 `floor(128MiB/(MB*4))`。
实际 helper 默认含 64 floats（256 bytes）guard：
`rows=min(N, floor((128MiB-256)/(MB*4)), optional_row_cap)`。

| N | MB | 旧 payload bytes | 旧 MiB | 默认新 rows | 新 allocation bytes（含256B guard） |
|---:|---:|---:|---:|---:|---:|
| 1024 | 256 | 1,048,576 | 1 | 1024 | 1,048,832 |
| 2048 | 512 | 4,194,304 | 4 | 2048 | 4,194,560 |
| 4096 | 1024 | 16,777,216 | 16 | 4096 | 16,777,472 |
| 8192 | 2048 | 67,108,864 | 64 | 8192 | 67,109,120 |
| 16384 | 4096 | 268,435,456 | 256 | 8191 | 134,201,600 |
| 32768 | 8192 | 1,073,741,824 | 1024 | 4095 | 134,185,216 |
| 65536 | 16384 | 4,294,967,296 | 4096 | 2047 | 134,152,448 |

64K 时需要 33 chunks；去掉 guards 的生产式规划可用 2048 rows / 32 chunks / 正好
128MiB。当前 standalone 将 guards 也计入上限，绝不偷偷多分配 256 bytes。
这个上限**仅限制 Q2 score device allocation**，不是总显存/RSS 上限。Q1 projection、
完整 iq/query、KV、selected 等仍增长。A/B 是两份独立 scratch，各自有界。

## 验证：无完整 host score matrix

`validated_score_rows` 对实际 Q0→Q3 输出保留的 sel/count 做流式验证：

1. 使用本 pipeline 的实际 iq/summary，按 chunk 重跑 scorer。
2. NumPy score reference 只分配当前 chunk，检查全部有效评分（bypass dense 不读）。
3. 对**每个 query**按自己的 GPU scores 做 deterministic exact selection，核对 IDs、
   counts、padding。dense 则检查 scratch 与本 chunk dispatch 前 bitwise 相同。
4. A/B 用两个逐行 iterator，在线计算 score delta、kth margins 和所有 changed IDs，
   不缓存所有 score rows。Q1 近 tie 导致的选择差异仍检查 margin ≤ 2*score delta。
5. Q3 仍用本路径自己的选择验证，`atol=rtol=2e-3` 不变。默认检查 ≤64 全部或32采样、
   边界以及**所有** A/B changed-selection rows；`--full-ref` 检查所有 query。
6. 原 replay、cache 未触及区域、全部输出/guards 的 exact 检查保留。

这保留了 manageable shapes 的强校验。代价是验证会再次评分，CPU reference 仍为
O(NT*MB)，但 score 工作空间只与 chunk 大小有关。可同时存在少量 chunk-sized host
数组，不保证整个 host RSS ≤128MiB。64K prefill 的完整模型几何参考可能很慢，不能
将 CLI 接受 64K 或纯 planner 通过视为该形状 GPU/端到端已验证。

## 实际 remote B580 测试结果

所有 GPU 测试只在 `openvino-ci-74@10.239.140.245:/mnt/river/qsa/cm_kernel`，
BatchMode SSH、`CM_FE_DIR=/mnt/river OPENBLAS_NUM_THREADS=1 timeout 300 .venv/bin/python`。

- `validate_q2_chunked.py --k 1`：52 configurations PASS。
- `validate_q2_chunked.py --k 0 512`：100 configurations PASS。
  合计 148 个不同配置（4 个长 row-boundary 配置重复运行）。最终代码再运行完整
  `validate_q2_chunked.py`，**148/148 PASS**。小 case 覆盖
  partition/DPAS × scalar/cooperative WG16 × dense bypass off/on × K0/1/512 ×
  row cap 1/3/15/16/17/31。每个配置在同一个 scratch 上 random→zero ties 两轮。
  每个 chunk exact valid score bits 对比**旧 ABI 的相同 kernel**；exact metadata；
  guard、未调度 output rows、最后未填满的 scratch rows 不变；再无中间同步重跑。
  mixed/zero-query/nonzero-past、visible2051/2052、complete-block2051/2052、
  65536 history / >4096 candidates 均包含。额外真实 row boundary2051/2052 两 scorer 通过。
- pure planner：NT0 dummy、partial16、sequence boundaries、1-row exact-fit、非整行预算向下
  取整、不足一行/guards/非法参数报错、64K 的旧4GiB/新≤128MiB 通过。
- `test_qsa_pipeline.py --compare --full-ref --score-row-cap 17`：全部9个 edge/continuity
  calls 两路径通过，包含真实历史和 K0/K1；全部 own-score IDs/counts exact。
- `--compare --validate-shapes --mode prefill --sizes 4096 --score-row-cap 31 --full-ref`：
  PASS；每路径133 chunks，scratch **127232 bytes**；全部4096 Q3参考通过。
  own-score reference 最大误差 baseline 3.8146973e-6、optimized 4.2915344e-6；
  Q3 最大误差 2.9575825e-4。Q1 导致的3个 A/B选择差异仍为 tokens2735/2791/3421，
  与历史结果一致，未弱化容差。
- `--compare --validate-shapes --mode decode --decode-past 65536 --decode-batches 1
  --score-row-cap 1 --full-ref`：PASS；真实65536-token scalar Q0/Q1历史及参考通过；
  每路径 scratch **65792 bytes**，own scores 最大误差1.9073486e-6、Q3 7.6182187e-6。
- 原 `validate_dense_bypass.py` 全6 geometry 与 instrumented score-read checks PASS；
  原 `validate_q2_cooperative.py` 全 suite（WG8/16、partition512/16/32/64、分布/计数）PASS。
- 原 score DPAS driver sizes17/31 both、partition driver sizes17/2052 both、topk driver
  sizes17/2052 both，`--check --iters 1 --no-flush` 全部 correctness OK。
- integrated optimized prefill4096 `--score-budget-mib 1` 自动规划255 rows、17 chunks、
  1044736 bytes，通过全部own-score/metadata及36行Q3参考；不依赖显式 row_cap。
  `--compare --validate-shapes --mode prefill --sizes 4096 --score-unchunked` 旧ABI两路径
  也通过，39行Q3参考包含所有3个选择变化行，scratch为16777472 bytes/路径。
- 本地26个Python文件 AST解析、三个新增/修改Python文件的 Pylance syntax diagnostics
  及编辑器错误检查全部通过；CM编译与执行以以上remote实际测试为准。
- 仅额外运行17-token、row_cap3、warmup1/sample1 的计时**功能冒烟**：6 chunks 的
  event数量、stage跨chunk求和、post-timing replay 通过；这些数值不用于性能结论。

**没有运行大 prefill benchmark，没有新的性能提升或大规模通过声明。**
`PIPELINE_OPTIMIZATION_RESULTS_CN.md` 保持不变，SHA256：
`f1026d682fae780ec1d4ab033c82a810c802c24dd54658917ebbe44c56ac65e9`。

## CLI 与下一轮大 benchmark 准备

从本地 `cm_kernel/` 仅同步以下 named paths；不传 `.env`、模型、其他 repo，不用 delete：

```bash
rsync -av -e 'ssh -o BatchMode=yes' qsa_score_chunked.py validate_q2_chunked.py test_qsa_pipeline.py README.md SCORE_SCRATCH_CHUNKING_CN.md openvino-ci-74@10.239.140.245:/mnt/river/qsa/cm_kernel/
rsync -av -e 'ssh -o BatchMode=yes' kernels/qsa_score_partition.cm kernels/qsa_score_tile_dpas.cm kernels/qsa_topk_finalization.cm kernels/qsa_topk_cooperative.cm openvino-ci-74@10.239.140.245:/mnt/river/qsa/cm_kernel/kernels/
ssh -o BatchMode=yes openvino-ci-74@10.239.140.245 'cd /mnt/river/qsa/cm_kernel && env CM_FE_DIR=/mnt/river OPENBLAS_NUM_THREADS=1 timeout 300 .venv/bin/python validate_q2_chunked.py'
```

常用参数：`--score-budget-mib 128`（默认；允许1..128），`--score-row-cap 31`（测试），
`--score-unchunked`（只有完整矩阵含guards不超过 budget 才允许，不能同时指定 row cap）。
`--validate-shapes` 不执行 benchmark；`--benchmark-only` 永远先做每个形状的严格验证。

以下是**已提供、尚未执行的大 benchmark 命令**，留给下一轮；应按尺寸逐次执行，确认
前一次结束、资源和 timeout 后再继续。64K 只修复了 Q2 scratch，不承诺整体内存/运行时：

```bash
ssh -o BatchMode=yes openvino-ci-74@10.239.140.245 'cd /mnt/river/qsa/cm_kernel && env CM_FE_DIR=/mnt/river OPENBLAS_NUM_THREADS=1 timeout 300 .venv/bin/python test_qsa_pipeline.py --optimized --benchmark-only --mode prefill --sizes 16384 --score-budget-mib 128 --warmup 100 --samples 20'
```

后续可逐次将 `--sizes` 改为32768/65536；`--flush` 只在整个 slice 前清 cache，
`--compare` 会额外运行慢 baseline 和所有差异行 Q3参考，`--full-ref` 会显著增加参考成本。
计时 JSON 输出实际 `score_scratch_bytes`（含guard）、`score_rows`、`score_chunks`；
stage时间是各chunk之和，GPU event总和不含dispatch gaps，host elapsed包含enqueue/finish。