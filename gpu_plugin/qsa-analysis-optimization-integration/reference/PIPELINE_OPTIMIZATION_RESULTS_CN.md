# Standalone CM 优化结果：Stage 1–5 与 Q0→Q1→Q2→Q3 集成验证

## 范围与结论

2026-09-10，仅修改 `cm_kernel/`，本阶段只实现 **exact dense score/top-k bypass**。
未实现 Q1 DPAS/key-only、cooperative Q2、decode split-KV 或新 Q3 tile；未修改插件、其他仓库或模型。

真实远程设备：`openvino-ci-74@10.239.140.245`，Intel Arc B580，160 EUs，设备报告最大频率 2900 MHz；
工作目录 `/mnt/river/qsa/cm_kernel`，解释器 `.venv/bin/python`，环境
`CM_FE_DIR=/mnt/river OPENBLAS_NUM_THREADS=1`。GPU 工作串行执行，没有本地 GPU 测试。

**结果：严格回归通过；1K/2K dense prefill 的 DPAS score→finalizer GPU event 总时间分别减少到基线的约 1/3.13、1/5.44。
这是 synthetic Q2 子流水线结果，不是 Q1→Q2→Q3 集成结果，更不是模型加速。**

## 1. 精确语义与开关

`n_complete = floor((abs_pos + 1) / r)`，且只有 `n_complete <= block_topk` 时跳过 scoring。
默认 `r=4, block_topk=512`：可见长度 2051 时仍是 512 个完整 block，2052 时是 513 个。
不能把 2048 当成该条件的边界，也不能把长上下文直接截成最早 512 blocks。

| 开关 | 默认 | 合同 |
|---|---:|---|
| `QSA_DENSE_BYPASS` | 1 | radix 在全选时直接输出升序 arange，不读取 scores；fused 在同一 dense 条件下不加载 indexer query、不扫描 summary、不写 score scratch |
| `QSA_DENSE_SCORE_BYPASS` | 0 | partition/DPAS 独立 score kernel 默认保持原 score 输出；仅流水线调用者显式设 1 才允许 dense rows 不写分数 |

- **完整基线**：`QSA_DENSE_BYPASS=0, QSA_DENSE_SCORE_BYPASS=0`。
- **协调优化**：score producer 和 finalizer 都用 `QSA_DENSE_BYPASS=1`，独立 score producer 额外用 `QSA_DENSE_SCORE_BYPASS=1`。
- 同一编译单元中 `QSA_DENSE_SCORE_BYPASS=1 && QSA_DENSE_BYPASS=0` 会触发 `#error`。
  该检查不能跨两个独立 program 检查 finalizer 的编译选项；调用者必须保证配对一致。
- 独立 score driver 不设置任何开关时，所有有效 score 仍写出。DPAS 原有 invalid/padded score-column stores 也保持不变。
- Fused 的 `block_scores` 本来就是 scratch：dense row 的内容不再有 score 输出保证。
- `k=0` 仍先返回 0；调用者按原合同写 `selected_counts` 和 `-1` padding。
  为严格遵循本阶段条件，K=0、n_complete>0 的 scoring 没有额外省略。
- DPAS 用最后一个有效 query 的 n_complete 判断整个 sequence-local tile；全 dense 时所有线程在 query staging/barrier 前统一返回。
  mixed tile 保留算术，只屏蔽 dense row 的 score stores。没有更改 DPAS tile geometry。
- **Q1 不跳过**：没有修改 prepare 调度或 indexer Q/raw K/summary 维护逻辑。新测试使用 synthetic Q2 输入，不是用测试替代 Q1 状态维护。
- **Q3 保持原实现**：没有改两个 sparse-attention kernel 或 `qsa_dpas_common.hpp`。公共头新增的 radix 分支不改变 Q3 attention 算法。

## 2. 严格验证

新增 `validate_dense_bypass.py`，默认总是检查，失败抛异常、进程非零；只有加 `--benchmark` 才计时。
未修改现有 Python driver；它们原有 `--check` 只打印的行为仍存在，故只作为数值 smoke test，不作为严格验收。

六组配置：

| 配置标签 | r | PA page | top-k |
|---|---:|---:|---:|
| r4-p16-k512 | 4 | 16 | 512 |
| r4-p16-k1 | 4 | 16 | 1 |
| r4-p16-k0 | 4 | 16 | 0 |
| r2-p16-k1 | 2 | 16 | 1 |
| r8-p32-k512 | 8 | 32 | 512 |
| r4-p64-k512 | 4 | 64 | 512 |

每组 58 个 query，混合空序列、零完整 blocks、past/nonzero history、短 prefill、decode、whole dense/sparse tile、跨 dense 边界 mixed tile、partial tile、打乱物理页。
每组跑随机 indexer 输入和全零 summary（严格同分）两种输入，共 12 个混合 batch；每种输入分别检查 partition、DPAS、fused 的 baseline/default/optimized。

严格逐元素比较：

1. selected IDs、counts、所有 `-1` padding，以及额外 dispatch 和输出末端 guard。
2. baseline 结果对照 CPU lexsort 的确定性规则（score 降序、ID 升序打破同分，最终输出 ID 升序），不是仅比较集合。
3. 默认独立 score 的整个 buffer 与基线一致；优化后的 sparse rows 与同 backend 基线一致。
4. dense rows 预先写 NaN / -12345 poison，score 后逐 bit 验证整行未写；finalizer 仍必须产生精确 metadata。
5. finalizer-only 额外覆盖 zero/equal/离散正负 ties/1 ULP near-ties，六配置共 24 组分布，每组 A/B。
6. K=0 时 IDs payload 长度为零，guard 不得改写，counts 必须为零；同时覆盖 K>0 但 n_complete=0。
7. 明确断言可见长度 2051/2052 的 n_complete 分别为 512/513，dense 标志分别为 true/false。

### 不读未写 dense scores 的证据

除了生产版本的 poison 检查，测试在内存中对 **实际 flatten 后的 finalizer/radix 源码**插桩：
两处 `scores[i]` load 均计数，每个 token 一个 counter（单 work-item 所有，无共享竞态）。源码匹配数量变化时测试会失败，防止漏插桩。
生产文件、ABI 和 benchmark kernel 不包含插桩。

| r4/p16/K512，58 rows | baseline dense reads | optimized dense reads | baseline/optimized sparse reads |
|---|---:|---:|---:|
| 随机输入 | 25770 | **0** | 72149 / 72149 |
| 全零 summary/ties | 25770 | **0** | 72056 / 72056 |

其余配置也全部通过零 dense read 检查，K=0 时全部读取计数为零。
基线 dense 正计数以及 sparse 正计数是插桩有效性的对照。
这些是插桩版本的动态源码 load 计数，不是生产 kernel 的硬件访存计数器；源码中 arange 返回位于任何 score load 前，配合未插桩 poison 测试建立合同证据。

### 已执行的命令

本地仅语法检查：
`python -c 'import ast, pathlib; p = pathlib.Path("validate_dense_bypass.py"); ast.parse(p.read_text(), filename=str(p)); print("AST parse PASS")'`。
Pylance 文件语法检查也返回无错误，编辑器诊断无错误。

同步仅枚举文件、保留目录层级，没有同步整个目录或 `--delete`：
`timeout 300 rsync -avR -e 'ssh -o BatchMode=yes' include/qsa_common.hpp kernels/qsa_score_partition.cm kernels/qsa_score_tile_dpas.cm kernels/qsa_score_topk_fused.cm kernels/qsa_topk_finalization.cm validate_dense_bypass.py openvino-ci-74@10.239.140.245:/mnt/river/qsa/cm_kernel/`。

所有远程 Python 命令通过以下包装执行：
`timeout 300 ssh -o BatchMode=yes openvino-ci-74@10.239.140.245 'cd /mnt/river/qsa/cm_kernel && timeout 300 env CM_FE_DIR=/mnt/river OPENBLAS_NUM_THREADS=1 .venv/bin/python <脚本和参数>'`。

| 脚本和参数 | 实测结果 |
|---|---|
| `validate_dense_bypass.py --configs r4-p16-k512` | 全部通过 |
| `validate_dense_bypass.py --configs r4-p16-k1 r4-p16-k0 r2-p16-k1 r8-p32-k512 r4-p64-k512` | 全部通过 |
| `validate_dense_bypass.py --configs r4-p16-k512 --benchmark --warmup 100 --samples 20 --sizes 1024 2048 4096` | 回归通过，12 条 BENCH 结果见下表 |
| `test_q2_score_tile_dpas.py --check --mode prefill --sizes 33 --iters 1 --no-flush` | OK，报告 max\|err\|=0.0000 |
| `test_q2_score_partition.py --check --mode decode --sizes 2050 2051 4096 --iters 1 --no-flush` | 三项均 OK，默认 dense rows 仍产出有效 score |
| `test_q2_score_topk_fused.py --check --mode decode --sizes 2050 2051 4096 --iters 1 --no-flush` | 三项均 OK |
| `test_q2_topk_finalization.py --check --mode decode --sizes 2050 2051 4096 --iters 1 --no-flush` | 三项均 OK |

decode 参数为 **past**，所以 2050/2051 对应可见长度 2051/2052。
旧 driver 的单次 smoke 计时不用于 A/B 结论。

## 3. 实测 A/B 性能

同一输入、相同布局、相同 dispatch；baseline 为两个宏均 0，optimized 为两个宏均 1。
每种 backend/size 各 100 次 warmup、20 个 samples/variant，偶数样本 A→B、奇数样本 B→A。
GPU profiling event 时间，单位 ms；分离路径记录 score 与 finalizer 两个 event 及其和；fused 记录一个 event。
每次测量单独 finish，不 flush（可选 `--flush` 支持 96 MiB 冲刷，但本表未启用）。
无插桩、无 host 数据复制计时；同一 in-order queue 保证 score→finalizer 依赖。
为了验证边界，dispatch 保留少量越界 work-items（token +3 / DPAS tile +1），A/B 一致。

默认模型 indexer Hidx=4、Di=128、r=4、K=512、PA16；prefill past=0，decode batch=1。

| 模式 | size | backend | dense rows / rows | 基线 mean | 优化 mean | 基线/优化 |
|---|---:|---|---:|---:|---:|---:|
| prefill | 1024 | DPAS + finalizer | 1024/1024 | 0.345286 | 0.110171 | 3.134x |
| prefill | 2048 | DPAS + finalizer | 2048/2048 | 0.835421 | 0.153692 | 5.436x |
| prefill | 4096 | DPAS + finalizer | 2051/4096 | 2.228634 | 1.774468 | 1.256x |
| prefill | 1024 | fused | 1024/1024 | 0.467776 | 0.108541 | 4.310x |
| prefill | 2048 | fused | 2048/2048 | 1.226448 | 0.151468 | 8.097x |
| prefill | 4096 | fused | 2051/4096 | 3.584796 | 3.112656 | 1.152x |
| decode | 1024 | partition + finalizer | 1/1 | 0.301207 | 0.043765 | 6.882x |
| decode | 2048 | partition + finalizer | 1/1 | 0.494676 | 0.018509 | 26.726x |
| decode | 4096 | partition + finalizer | 0/1 | 0.728015 | 0.714807 | 1.018x |
| decode | 1024 | fused | 1/1 | 0.291968 | 0.042854 | 6.813x |
| decode | 2048 | fused | 1/1 | 0.477286 | 0.017947 | 26.594x |
| decode | 4096 | fused | 0/1 | 0.906302 | 0.918005 | **0.987x** |

分离路径的阶段 mean：

| 模式/size | 基线 score | 优化 score | 基线 finalizer | 优化 finalizer |
|---|---:|---:|---:|---:|
| prefill/1024 | 0.006859 | 0.000999 | 0.338427 | 0.109171 |
| prefill/2048 | 0.019859 | 0.001416 | 0.815562 | 0.152275 |
| prefill/4096 | 0.064526 | 0.060895 | 2.164109 | 1.713572 |
| decode/1024 | 0.106265 | 0.000583 | 0.194942 | 0.043182 |
| decode/2048 | 0.211729 | 0.000578 | 0.282947 | 0.017932 |
| decode/4096 | 0.212156 | 0.207052 | 0.515859 | 0.507755 |

总时间 median / P95：

| 模式/size/backend | 基线 median | 优化 median | 基线 P95 | 优化 P95 |
|---|---:|---:|---:|---:|
| prefill/1024/dpas | 0.345311 | 0.110155 | 0.345624 | 0.110415 |
| prefill/2048/dpas | 0.835415 | 0.153697 | 0.836260 | 0.153859 |
| prefill/4096/dpas | 2.228645 | 1.774010 | 2.233135 | 1.778050 |
| prefill/1024/fused | 0.467552 | 0.108541 | 0.469176 | 0.108864 |
| prefill/2048/fused | 1.226562 | 0.151458 | 1.228614 | 0.151770 |
| prefill/4096/fused | 3.582551 | 3.105364 | 3.593843 | 3.162395 |
| decode/1024/partition | 0.301197 | 0.043750 | 0.301567 | 0.043854 |
| decode/2048/partition | 0.494791 | 0.018541 | 0.495108 | 0.018645 |
| decode/4096/partition | 0.728280 | 0.714791 | 0.729171 | 0.715520 |
| decode/1024/fused | 0.291979 | 0.042812 | 0.292395 | 0.042916 |
| decode/2048/fused | 0.477187 | 0.017916 | 0.478125 | 0.018020 |
| decode/4096/fused | 0.906250 | 0.917968 | 0.907500 | 0.919687 |

## 4. 风险、限制与回退

- **配对开关是硬合同**：不能让 bypass score producer 搭配 baseline finalizer；本阶段没有运行时跨 program 验证。
- 全选时仍写 IDs 和 padding，不是零成本；这两种写循环、编译路径和 GPU 状态不同，不能假设不同可见长度时延单调。
- 4K prefill mixed tile 仍有 dense-column 算术；本阶段不改变 cooperative scoring、radix 或调度粒度。
- 4K decode 全 sparse，无 dense 工作可删除。本轮 fused **回退约 1.29%**（0.906302→0.918005 ms），没有归因为某个未经 profiling 证实的原因；partition 的约 1.8% 差异也不能算 dense 算法收益。
- 一轮 warm-cache 交替测量不是跨进程、跨 GPU 状态的长期统计；不使用旧报告的 cold-cache 数字计算倍率。
- poison 测试使用 NaN 检测未写行，不意味着对有效 score 的 NaN 排序作新保证。
- Q1/raw K/summary、Q3/output gate 的真实端到端链路没有在新脚本中运行；保持源文件不变不等于新增集成验证。
- 回退使用两个宏均 0。共享头的开关面向本目录 standalone 编译；没有宣称已在插件 batched codegen 中接通。

## 5. 修改清单与未改动证明

本阶段恰好新增/修改以下七个文件：

1. `include/qsa_common.hpp`：宏默认与全选 arange。
2. `kernels/qsa_score_partition.cm`：显式 opt-in dense early exit。
3. `kernels/qsa_score_tile_dpas.cm`：全 dense tile early exit 与 dense row store skip。
4. `kernels/qsa_score_topk_fused.cm`：dense query scoring skip。
5. `kernels/qsa_topk_finalization.cm`：配对合同注释（逻辑经公共 radix 生效）。
6. `validate_dense_bypass.py`：新严格回归和可选 benchmark。
7. `PIPELINE_OPTIMIZATION_RESULTS_CN.md`：本报告，创建前已确认不存在。

修改前后本地与远程以下 SHA256 均保持一致：

| 文件 | SHA256 |
|---|---|
| `kernels/qsa_prepare.cm` | `44b64e1d8f34a623f0a714ea71fa5f4b637cd33abed4417a8ff7d71b4f0c77cf` |
| `kernels/qsa_sparse_attention.cm` | `a731a8ac4367513101b13ec4a3b96a63d4f9898463dfc2328a75f36c45b15081` |
| `kernels/qsa_sparse_attention_dpas.cm` | `d94f4063219df03eac97f322a78e0162f99db2c4cd2e0fb67a0f71d913f32d8e` |
| `include/qsa_dpas_common.hpp` | `2c4c2da379c2f55076ed9922c2628041216115fe7e6cfaa3afa1783586cf551c` |

---

# Stage 2：Q1 DPAS 投影 + FP32 prepare 后处理

2026-09-10。以上 Stage1 内容原样保留；本节仅记录随后完成的 **Q1 Stage2**。
仅修改 standalone `cm_kernel/`，未接入 OpenVINO、未实现下一阶段、未改变 Q2 dense bypass 或 Q3。
远端设备、目录、解释器与 Stage1 相同；所有 GPU 编译、验证、计时均在 B580 上串行执行。

## S2.1 实现与调用合同

本阶段新增/修改六个文件（包括本报告）：

| 文件 | 用途 |
|---|---|
| `kernels/qsa_project_dpas.cm` | 实际 `cm_dpas<HF,HF,8,8,float>`，M=8 输出通道、N=16 token、K=16；FP32 累加并写 `[tokens,J]` |
| `kernels/qsa_prepare.cm` | `QSA_PREPROJECTED` 默认 0；设为 1 时首参数新增 `const float* projected`，其余参数原序不变 |
| `qsa_prepare_dpas.py` | `PrepareDPAS` 编译并依次 enqueue 投影与 prepare，无内部 scratch 分配、copy 或 finish |
| `test_q1_prepare.py` | 新增显式 `--dpas`；不加参数仍为原单-dispatch scalar baseline，既有参数保持有效 |
| `validate_q1_dpas.py` | 严格异常退出的数值/guard/缓存验证、实际 scalar 投影捕获、完整 Q1 A/B |
| `PIPELINE_OPTIMIZATION_RESULTS_CN.md` | 仅追加本节 |

- 使用原 `qsa_dpas_common.hpp` loaders，未修改共享头：A 为 `[8 output rows,16 K]` 的 weight 普通 2D load；
  B 为 token-major activation 的 transposed d32 load，直接得到 VNNI `[K/2,N,2]`。
- 输出通过 `transpose_8x16` 回到 token-major。每个 vector store 为 8 floats，地址为 **byte offset**。
  描述符 height 相对于已经偏移到 tile 起点的 base；最后不足 16 个 token 的 load 用真实剩余高度，store 显式限制有效行。
  每个 2D message 的行宽仅 32B，没有超过 64B message-row 上限。
- 投影 dispatch 为 `[J/8,ceil(tokens/16),1]` / `[1,1,1]`。每个线程只有一个 8×16 FP32 accumulator。
  `j0>=J` 和 `t0>=tokens` 显式返回；**不支持 J 非 8 倍数的尾输出 tile**，有 compile-time assert 和 host 拒绝。
- prepare 的 preprojected 分支不分配/加载 activation SLM、不计算 scalar dot，只从 FP32 scratch 填 `slm_proj`。
  RMSNorm、NEOX RoPE、position_ids、raw K cache、完成块 summary、旧 raw cache 读取的实现不变。
  **新 raw K 仍以 FP32 留在 SLM 参与 pooling**；写入 half cache 后绝不重读它来替代 FP32 值。
- 同一个 in-order queue 上连续两个 dispatch 建立 producer→consumer 依赖；若移到 out-of-order queue，调用方必须另加事件依赖。
  运行时 scratch 由调用方提供、维持生命周期，大小 `tokens*J*4` bytes（1K/2K/4K 为 2.5/5/10 MiB）。
  不支持 qbase、任意 tensor strides 或运行时权重转置；token 下标沿用原 prepare 的从 0 开始的全 batch 布局。
  投影 tile 可以跨 sequence，因为投影本身没有位置状态；后处理仍由 sequence/block prepare_map 决定。

### 支持配置与默认/回退

本轮主配置 `D=2560,J=640,Hidx=4,Di=128,rot=32,r=4,PA=16,prepare WG=16`。
另验证 `D=96,J=144,Hidx=2,Di=48,rot=16,r=2,PA=32`。
硬件要求 CM DPAS、LSC 2D、512-bit GRF（N=16，B580 已验证）。
投影要求 `D>=32 && D%32==0`（surface 宽度至少 64B，row pitch 为 64B 倍数）、`J%8==0`；
prepare 保持 `Di%16==0`、`Hidx+1<=WG`、`PA%r==0` 及原 rotary/SLM/索引范围约束。
这些维度条件不是对任意大 shape/任意 GPU 的支持承诺。无自动 padding 或自动跨设备 fallback。

**默认仍为 scalar**：不设置 `QSA_PREPROJECTED`、不传 `--dpas` 即可回退；新增 helper 仅显式调用。
本轮 batch=1 DPAS 也测得更快，但没有据此强制 decode 自动切换或推断所有小 batch/硬件都会更快。

## S2.2 正确性与数值误差

两次完整运行（no-flush/cold benchmark 前各一次）均通过：

1. 两个随机种子分别验证 token 数 1、15、16、17、59，以及全空 batch；59-token mixed batch 包含空序列、
   past=3/15/34、跨 compression block/PA page、非零 global token begin、打乱物理页、partial tile。
2. position_ids 故意加 7，不同于 cache absolute position；query RoPE 用 position_ids，summary RoPE 仍用 block start。
3. 额外 whole-zero activation、替代 D/Di/r/PA 配置；不支持的 D=48、Di=127 在 host 编译前拒绝。
4. GPU scalar 投影对照来自 **实际 baseline flatten 源码** 的 test-only FP32 capture，不是另写一个 scalar GEMM。
   小 case 同时比较 FP64 CPU GEMM；1K/2K/4K 大 case 比较 FP32 BLAS（输入始终是同一 half 数据）。
   benchmark 使用未插桩 baseline，不带 capture store。
5. iq/raw/summary 同时比较 baseline 与 NumPy reference；reference 复用 `qsa_ref` norm/RoPE，
   在新 driver 内正确处理旧 raw cache 与新 FP32 raw K 混合 pooling，未修改共用 `qsa_ref.py`。
   原 `prepare_ref` 对跨 past block 的旧 raw K 没有参数支持，因此不用它作为此类 case 的 oracle。
6. 对输出末端 32-element guard、额外越界投影 tile/prepare WG、旧 raw cache、未完成/未触及 summary 做精确检查。
   全空 batch 为 host no-op。首次测试遇到 clops 不支持零字节 copy，仅修复新验证脚本的空输入占位分配；CM kernel 无编译/数值失败。
7. 单独向实际 prepare 后处理提供 FP32 raw K `[1.0001,-1,0,0]` 的消去输入，
   通过 FP32 reference，且 half-cache 重读的错误替代方案会使 summary 最大差 **0.045654297**；测试明确排除该错误。
8. 每种 benchmark shape 计时前也完整检查 iq/raw/summary 和投影，不以计时替代正确性。

最大绝对误差（两轮结果一致，非 bit-exact）：

| 比较 | 最大绝对误差 |
|---|---:|
| 小 case DPAS FP32 projection vs FP64 oracle cast FP32 | 2.384186e-6 |
| 小 case DPAS FP32 projection vs 实际 scalar capture | 1.668930e-6 |
| 大 case DPAS FP32 projection vs FP32 BLAS | 4.291534e-6 |
| 最终 half iq / raw / summary vs baseline，各自最大值 | 0.001953125 / 0.001953125 / 0.001953125 |
| 最终 half iq / raw / summary vs reference，各自最大值 | 0.00390625 / 0.00390625 / 0.001953125 |

断言阈值：projection `atol=3e-5,rtol=2e-5`；最终输出 `atol=8e-3,rtol=2e-3`；
FP32 pooling 消去用例 `atol=rtol=1e-3`；未触及缓存与 guard 精确相等。
DPAS 和 scalar dot 的累加次序不同，接近 half 舍入边界时会落到相邻表示。
**这不是 top-k bit-exact 承诺**：极近分数/ties 的 sparse top-k 可能变化；本阶段未执行 Q1→Q2→Q3/模型质量验证。

### 已执行的远程验收

统一包装：`timeout 300 ssh -o BatchMode=yes openvino-ci-74@10.239.140.245 'cd /mnt/river/qsa/cm_kernel && env CM_FE_DIR=/mnt/river OPENBLAS_NUM_THREADS=1 timeout 300 .venv/bin/python <脚本和参数>'`。

| 脚本和参数 | 结果 |
|---|---|
| `validate_q1_dpas.py --benchmark --warmup 100 --samples 20 --sizes 1024 2048 4096` | 严格验证全通过，6 条 no-flush A/B |
| `validate_q1_dpas.py --benchmark --flush --warmup 100 --samples 20 --sizes 1024 2048 4096` | 严格验证再次全通过，6 条 cold A/B |
| `test_q1_prepare.py --check --sizes 17 33 --mode both --iters 1 --no-flush` | 默认 ABI，4 项 OK |
| `test_q1_prepare.py --dpas --check --sizes 17 33 --mode both --iters 1 --no-flush` | 显式两-dispatch driver，4 项 OK |
| `validate_dense_bypass.py` | Stage1 全六配置、三 backend、poison、精确 metadata、动态 reads、ties 测试通过 |

旧 driver 的 `--check` 仍只打印，因此严格验收以新 validator 异常退出为准；smoke 的单次时延不用作 A/B。
本地只运行 AST/Pylance 语法与编辑器诊断，未执行 GPU。原 `.env` 已存在，未读取/同步秘密。

## S2.3 完整 Q1 A/B（两次 dispatch，单位 ms）

每个 shape/variant 100 次 warmup、20 samples，偶数 A→B、奇数 B→A，输入相同，输出/scratch 独立。
baseline 一个 event；DPAS 两个 event，顺序为 projection、prepare；无 capture 插桩，无额外越界 dispatch。
GPU total 是两个 event duration 的和，**不包含 event 间间隙或 host launch 开销**；另列 host enqueue→finish 时间，
包括两次 launch、queue gap 与同步，但不含分配、CPU reference、数据传输和编译。

### No-flush（复用输入，不主动冲刷缓存）

| 模式 | size | baseline GPU | DPAS projection | DPAS prepare | DPAS GPU total | 倍率 | baseline host | DPAS host |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| prefill | 1024 | 5.032906 | 0.153651 | 0.028375 | 0.182025 | 27.65x | 5.046884 | 0.203383 |
| prefill | 2048 | 9.491307 | 0.349760 | 0.063942 | 0.413702 | 22.94x | 9.504941 | 0.433616 |
| prefill | 4096 | 18.879968 | 0.726901 | 0.162698 | 0.889598 | 21.22x | 18.891033 | 0.912645 |
| decode B1 | 1024 | 0.177375 | 0.015473 | 0.001364 | 0.016838 | 10.53x | 0.191112 | 0.034700 |
| decode B1 | 2048 | 0.177744 | 0.015536 | 0.001344 | 0.016879 | 10.53x | 0.191603 | 0.034092 |
| decode B1 | 4096 | 0.177401 | 0.015463 | 0.001375 | 0.016838 | 10.54x | 0.191322 | 0.034395 |

### Cold protocol（每个完整 Q1 之前单独 flush 96 MiB）

flush 本身不计时；**投影与 prepare 之间不 flush**，保留真实 producer→consumer scratch 复用。
“cold”指此冲刷协议，不是硬件计数器证明的 100% cache miss。

| 模式 | size | baseline GPU | DPAS projection | DPAS prepare | DPAS GPU total | 倍率 | baseline host | DPAS host |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| prefill | 1024 | 5.215718 | 0.187109 | 0.031864 | 0.218973 | 23.82x | 5.230817 | 0.239334 |
| prefill | 2048 | 9.668244 | 0.373942 | 0.068370 | 0.442312 | 21.86x | 9.682000 | 0.462774 |
| prefill | 4096 | 19.012703 | 0.742416 | 0.163255 | 0.905671 | 20.99x | 19.023809 | 0.926431 |
| decode B1 | 1024 | 0.354791 | 0.021338 | 0.002874 | 0.024213 | 14.65x | 0.369710 | 0.042875 |
| decode B1 | 2048 | 0.354619 | 0.021515 | 0.002947 | 0.024462 | 14.50x | 0.369336 | 0.042849 |
| decode B1 | 4096 | 0.355296 | 0.021323 | 0.003072 | 0.024395 | 14.56x | 0.370859 | 0.042853 |

prefill size 为新 token 数，past=0；decode size 为 past，query=1（这三个 decode case 均未完成新 summary block；
完成块 decode 的正确性由 past=3 测试覆盖，未为它单列 A/B）。各行是 mean，不与旧报告中不同 cache 协议数值混算。

GPU total median / P95（baseline → DPAS）：

| 协议/模式 | size | median | P95 |
|---|---:|---|---|
| no-flush/prefill | 1024 | 5.032708 → 0.182187 | 5.034744 → 0.185577 |
| no-flush/prefill | 2048 | 9.491614 → 0.413541 | 9.496979 → 0.420765 |
| no-flush/prefill | 4096 | 18.879270 → 0.889270 | 18.918927 → 0.900785 |
| no-flush/decode | 1024 | 0.177344 → 0.016874 | 0.179687 → 0.016879 |
| no-flush/decode | 2048 | 0.177760 → 0.016874 | 0.179900 → 0.016979 |
| no-flush/decode | 4096 | 0.177447 → 0.016874 | 0.179583 → 0.016874 |
| cold/prefill | 1024 | 5.215677 → 0.218957 | 5.217817 → 0.221254 |
| cold/prefill | 2048 | 9.668437 → 0.443958 | 9.669822 → 0.445989 |
| cold/prefill | 4096 | 19.013021 → 0.907551 | 19.015213 → 0.910962 |
| cold/decode | 1024 | 0.354843 → 0.024166 | 0.355646 → 0.024483 |
| cold/decode | 2048 | 0.354635 → 0.024478 | 0.355322 → 0.024800 |
| cold/decode | 4096 | 0.355312 → 0.024322 | 0.355937 → 0.024703 |

结论限定为当前 standalone **完整 Q1** 的改善，而不是整个 attention/model 加速。
本轮没有继续调 projection tile、合作 Q2、decode split-KV 或 Q3；保留后续分阶段决策。

## S2.4 同步与未改动检查

只 rsync 以下枚举文件，未同步整个目录、父 config、`.env` 或任何 OpenVINO 文件：
`kernels/qsa_prepare.cm kernels/qsa_project_dpas.cm qsa_prepare_dpas.py validate_q1_dpas.py test_q1_prepare.py PIPELINE_OPTIMIZATION_RESULTS_CN.md`。
源码/driver 的本地和远端 SHA256 逐项一致：

| 文件 | Stage2 SHA256 |
|---|---|
| `kernels/qsa_prepare.cm` | `9cf104df18ac6c84a9ba48d15fc51345755d003181acf5796b204241f872077b` |
| `kernels/qsa_project_dpas.cm` | `b0f6606a8663b7176bf38a14fe7f9567b04684a1d699408a1113e6bac019df90` |
| `qsa_prepare_dpas.py` | `c2d3902572f459617b7c58805603da48bac3595083faa53f5c057adcc1e27e52` |
| `validate_q1_dpas.py` | `241a02ad6eb7d1677fe05288d9ca7164973d3c2964a490445e9b5038f23cafa3` |
| `test_q1_prepare.py` | `201e41053a9cd40c84ce9c42a43de10ca8304cf2f7505d67ae632c24627ddbf2` |

两个 Q3 kernel 与 `include/qsa_dpas_common.hpp` 仍与上面 Stage1 表中的 SHA256 完全相同。
Stage1 的六个实现/验证文件未编辑；其本地/远端校验一致，且完整严格回归再次通过。
Stage1 中 prepare 的旧 SHA 是该历史阶段的记录；Stage2 的 prepare SHA 以本节为准。

---

# Stage 3：Standalone cooperative Q2 exact top-k + decode partition 粒度调优

2026-09-10。本节追加前，整个 Stage1/2 报告 SHA256 为
`6d48571928f262c4bcbaff3bb6b193ce372d2d66ecc8f6928f78a2112bc60d19`；上文原样保留。
只新增 standalone Q2 文件，**没有编辑任何现有 kernel、共享头或 driver，没有修改 Q1/Q3/OpenVINO**。
没有 Stage4、模型端到端、Q1 DPAS rounding 或 attention 精度变更。

**结论：exact correctness 全通过。B580 B1 sparse decode 推荐显式启用 WG16 + partition16；
相对已有 Stage1 dense-bypass scalar Q2，no-flush past 4K/16K/64K 完整 score→topk 分别快
10.45x / 15.23x / 15.41x。全 dense 1K/2K prefill 可用 WG8 加快 metadata 写出；
mixed 4K prefill cooperative 反而更慢，保留 scalar。没有安装全局 auto dispatch，没有更改既有默认。**

## S3.1 文件、算法与 ABI

本阶段恰好新增三个代码文件、追加一个报告：

| 文件 | 变更 |
|---|---|
| `kernels/qsa_topk_cooperative.cm` | 新增，一个 WG 拥有一条 score row，精确 cooperative radix + 有序压缩 |
| `qsa_topk_cooperative.py` | 新增 `TopKFinalizer`、`PartitionScorer`，显式 opt-in，原 kernels 不变 |
| `validate_q2_cooperative.py` | 新增严格测试、实际 score→topk 流水线、相同 GPU score buffer 的 finalizer-only A/B、partition 消融 |
| `PIPELINE_OPTIMIZATION_RESULTS_CN.md` | 只追加 Stage3 |

新 CM kernel 与 `qsa_topk_finalization.cm` **参数及顺序完全相同**：scores、past_lens、subsequence_begins、
selected_blocks、selected_counts、num_tokens、num_seqs、max_blocks。
dispatch 为 `[num_tokens*WG,1,1] / [WG,1,1]`，WG 支持 8 或 16；scalar 仍为每 row 一个线程。

### 4×8-bit cooperative radix

1. 每个 CM thread 扫描一个连续 block-ID 范围，长度为 `ceil(n_complete/WG)`，末线程可为空。
  私有 `vector<uint,256>` 直方图只统计已解析高位 prefix 相符的 key。
2. 私有 hist 写到 SLM 自己的 1024B row；local fence + barrier 后 leader 逐 row 向量相加。
  leader 从 bin255 向下找全局 kth bin，更新 prefix 和 remaining，通过 16B SLM state 广播。
3. 第二个 local fence + barrier 保证 hist 读完才能复用；下一轮 hist barrier 同时保护上一轮 state 读取。
  4 轮共解析完整 32-bit ordered key，复用 **原始** `qsa::float_to_ordered_key`，没有重算 score。
4. 所有线程再完整扫描自己的连续范围，统计 `key>threshold` 与 `key==threshold`。
  leader 按 tid/ID 递增顺序分配 equal quota（总共只取全局 remaining 个），并计算最终输出前缀偏移。
5. 第二次完整扫描按 ID 递增 emit；不同线程写互不重叠的连续输出段，因此整个输出 ID 升序。
  阈值上方的全部 block 都入选，阈值同分按最小 block ID 补足；不使用候选数组、局部 top-k 或 bounded-candidate 截断。

SLM：`WG*1024 + WG*16 + 16` bytes，WG8 为 **8336B**，WG16 为 **16656B**，都小于 64KiB。
hist 的元素为 uint32，没有 8-bit count 溢出；没有 WG 内 global peer-write/read 回环或 SLM atomics。
一次 sparse selection 为 4 次 hist 扫描 + count + emit，即 6 次 score row 扫描；时间不是常数，仍随 n 增长。

### Dense / K0 / padding 合同

- counts 为 `min(K,n_complete)`；所有剩余 `[count,K)` entries 必须写 `-1`。
- K0 / n_complete0 在任何 score load 与 barrier 前统一返回；K0 只写 counts，不写 IDs payload。
- `QSA_DENSE_BYPASS=1` 且 `n_complete<=K` 时直接并行写 arange，不读取 scores、不使用 radix/SLM barrier。
  dense 的 metadata 写出并非零成本。kernel 资源仍由编译后的整个 program 决定，不能声称 dense 消除了 SLM 资源预留。
- 关闭 dense bypass 的 cooperative 版本也做过精确验证；此时 producer 必须写全所有有效 scores。
- 分离 scorer 仍仅在显式 `QSA_DENSE_SCORE_BYPASS=1` 时不写 dense rows。
  原共享头、partition kernel 与 DPAS score kernel 的 Stage1 行为全部保留。

## S3.2 支持范围、显式选择与回退

`TopKFinalizer(cfg)` 默认 `cooperative=False`，走原 scalar finalizer；只有明确指定
`cooperative=True,wg=8/16` 才使用新 kernel。`PartitionScorer(cfg)` 默认仍是 partition512、
`dense_score_bypass=False`；新 helper 不影响原测试脚本的默认值。
两者 enqueue 内没有内存分配、copy 或 finish；caller 提供 scratch/输出并管理生命周期。
空 token batch 是 host no-op，不为零字节 tensor 分配制造 clops copy 错误。

实测后的**人工 opt-in 建议**（不是已安装的自动路由，也不是对未测 shape 的速度承诺）：

| 条件 | finalizer | score producer | 决策 |
|---|---|---|---|
| B580，默认模型配置，B1 decode，past=4096/16384/65536 | WG16 | partition16 | 三次 sweep 都有明显 pipeline 收益，可显式启用 |
| 默认配置，past0 prefill=1024/2048，全部 rows dense | WG8 | 原 DPAS score + coordinated dense bypass | 加速的是 arange/padding，而不是 sparse radix |
| 默认配置 mixed prefill=4096（2051 dense、2045 sparse） | **scalar** | 原 DPAS score + coordinated dense bypass | WG8 慢约 4%，WG16 慢约 37%，明确回退 |
| 其他设备、未测 batch/row 长度、其他配置或无性能证据 | **scalar** | 原配置/512，或调用者已独立验证的配置 | 不根据单个 B1 结果猜测自动阈值 |

全选判断必须用每个 token 的 `(abs_pos+1)//r<=K`，不是固定 token 常量 2048。
混合 batch 不能仅看最后一个全 batch token 来判断是否全 dense；本次不实现 host row 分桶或两种 finalizer 混合 dispatch。
scalar 是保留的显式 fallback，尤其面向大量较短 sparse rows；本次没有偷偷以 cooperative 替换它。

decode 只修改 `QSA_PARTITION_BLOCKS` 编译参数及 grid 分区数，query load / dot / per-head ReLU /
head reduction / scale 的顺序一行未动。**这是 partition 粒度调优，不是 cooperative scoring 实现**。
16 在 4K/16K 最好；64K 时 16/32 差异很小、会反转，不硬编码 64K→32。
选 16 是这一测量范围的统一显式配置；512 始终可回退。没有宣称 16 是所有长度/所有 batch 的全局最优。

正确性支持与性能选择要分开：WG8/16、K0/1/512、r4/PA16，以及补充 r8/PA32/K1 已验证；
16384 complete blocks 的 benchmark 也精确通过。无固定 4096 candidate cap；并非无限索引范围承诺。
caller 必须提供有效 sequence metadata、`max_blocks>=n_complete`、足够大的 score/output buffers，
并遵守现有 uint32 地址计算范围；DPAS score row stride 仍须按原 helper pad 到 8。
有限 FP32 的排序行为完全继承 scalar ordered-key（包括其区分 +0/-0 的既有行为）；
真实 Q2 的零分为 +0。没有新增有效 NaN scores 的数值排序合同，poison NaN 只用于检测 dense 未读/未写。
只在支持 CM/SLM 的 B580 实机验证，没有新增其他设备的自动编译回退。
score→finalizer 依赖使用同一 in-order queue；out-of-order queue 需要 caller 加依赖。
producer/finalizer 的 dense 开关仍是跨 program 配对合同，helper 不读取 GPU metadata 去验证不匹配的调用。

## S3.3 严格正确性与 Stage1 回归

新增测试默认执行断言，任何失败进程非零。第一次远端 CM 编译/全部测试即通过，**没有编译或数值失败修复循环**。
本地只做 AST、Pylance syntax 与编辑器 diagnostics，不执行 GPU。

主测试对每个 K0/1/512 使用 **42-token mixed batch**，包含：

- 空序列、n_complete0、短 prefill、17-token partial/mixed tile、非零 global token begin；
- 可见 token 长度 **2051/2052** 对应 n_complete **512/513**，明确断言；
- 另有 n_complete **2051/2052、4097、8193**，避免把 token boundary 与 block 数混淆；
- nonzero past、shuffled physical pages、跨 PA page、最后尾 tile。

每个 K 跑随机 indexer 输入、全零 summary 两种真实 scoring 分布，分别对 partition 和 DPAS：

1. 原 scalar + 全分数 baseline 对 CPU lexsort，检查 IDs、counts、所有 `-1` padding 和输出末端 32-element guard。
2. WG8/16 finalizer 直接消费 **同一 baseline GPU score tensor**，不是重新计算后假定两者分数接近。
3. 协调 dense-bypass 的实际 score→cooperative pipeline 对 baseline 全输出精确相等。
4. partition16/32/64 分别开启/关闭 score bypass，**整个 score buffer 逐 bit 比较**；
  sparse rows 与该 backend baseline 一致，dense rows 保持 NaN/-12345 poison。
  DPAS 同样只与它自己的 baseline 比较，完全不混入 Q1 DPAS rounding 差异。
5. 额外 finalizer-only 分布：random normal、全零、全0.5、正负离散 ties、1 ULP near-ties、
  `k-1` 个较高 score + 巨大的 kth-threshold tie，以及随 ID 单调增大的 scores（大 row 的胜者位于4096之后）。
  所有分布均比较 WG8/16 × dense bypass on/off，输出不是只比较集合。
6. 多 dispatch 时额外发出 3 个越界 WG，guard 不变；K0 的 IDs payload 长度为零，仍检查 guard。
7. 补充 r8/PA32/K1 的 mixed/scattered-page pipeline 和全空 host enqueue。

### 实际源码 load 插桩（仅测试，不进入计时 kernel）

只在 flatten 后的 cooperative kernel body 内替换三处 `scores[i]`，源码 site 数不为3就失败；
每个 CM thread 独占自己的 read counter，无原子/跨线程计数竞态。
WG8 和 WG16 分别检查每行 counter sum，而非仅总数：

| K | dense/K0 reads | sparse reads 总数（WG8 与 WG16 相同） |
|---|---:|---:|
| 0 | **0** | 0 |
| 1 | **0** | 206196 |
| 512 | **0** | 187602 |

每个 sparse row 精确为 `6*n_complete`；额外越界 WG counters 全零。
这证明实现扫描所有输入而非候选截断，也验证 dense/K0 未读未写 scores。
这是动态源码 load 计数，不是硬件 cache/带宽 counter。

原 `validate_dense_bypass.py` **未修改并完整通过**：六配置、12 个真实 mixed batches、
partition/DPAS/fused、baseline/default/optimized、poison、动态 reads 与24组独立分布回归全部通过。
Q1/Q3 源码保持不变；本阶段没有重跑 Q1 完整数值 suite 或 Q3 attention，因此不把源码 hash 当作端到端验证。

## S3.4 测量协议与复现记录

远端 GPU：`openvino-ci-74@10.239.140.245`，Intel Arc B580，160 EUs，报告最大频率2900MHz。
只在 `/mnt/river/qsa/cm_kernel`，`CM_FE_DIR=/mnt/river OPENBLAS_NUM_THREADS=1`，`.venv/bin/python` 执行 GPU。
所有 SSH 使用 `-o BatchMode=yes`，外层 SSH 与远端 Python 都限制 `timeout 300`；GPU 工作串行执行。
只 rsync 明确枚举的新三个源文件，最后单独同步本报告，无整个目录覆盖、无 `--delete`、无 `.env` 同步。
已有 `.env` 存在，未读取或修改秘密。

实际已执行（每条均成功）：

| 远端 Python 参数 | 结果 |
|---|---|
| `validate_q2_cooperative.py` | 全严格验证通过 |
| `validate_q2_cooperative.py --benchmark-only --warmup 100 --samples 20` | 首轮 no-flush，全6 shapes、12组结果 |
| `validate_q2_cooperative.py --benchmark-only --flush --warmup 100 --samples 20` | cold，全6 shapes、12组结果 |
| `validate_q2_cooperative.py --benchmark-only --modes decode prefill --warmup 100 --samples 20` | no-flush 倒序复测，全6 shapes、12组结果 |
| `validate_dense_bypass.py` | 原 Stage1 全部回归通过 |

`--benchmark-only` 只跳过小 shape 的 edge suite；**每个 benchmark variant 仍先实际运行 baseline、
对 CPU oracle 和完整 GPU metadata/score buffer 做严格比较**。不是盲计时。

- prefill=1024/2048/4096，past0，原 DPAS score backend；decode B1 past=4096/16384/65536，partition backend。
- 性能 baseline 已有 Stage1 dense bypass：producer/finalizer 都协调开启。不是把 Stage1 的收益重复算进 Stage3。
- 每个 scope/shape/variant 100 次 warmup，20 samples；偶数迭代 variants 正序、奇数反序，
  每对 A/B 的先后顺序都交替。prefill 比较 scalar/WG8/WG16；decode 另加 scalar/WG16 × partition16/32/64，
  共9 variants，不是先独立跑完整 A 再完整 B。
- 完整 pipeline 记录 score 和 topk 两个 GPU event；finalizer-only 单独使用同一 baseline GPU score tensor。
  使用 production 无插桩 kernel、无额外越界 dispatch。
- GPU total 为 event duration 之和，不包括 event 间 queue gap/host launch；另记录 host enqueue→finish。
  分配、CPU reference、编译、host 数据拷贝不计时。
- no-flush 不主动冲刷；cold 每个测量项前单独 flush96MiB 并 finish；**score→topk 之间不 flush**。
  finalizer-only 的 cold 是直接对 score buffer 冲刷后的独立测试，不能与 pipeline 中的 topk event 混为一谈。
  warmup 不 flush；cold 仅指该协议，不是硬件证明100% cache miss。

原始 stdout（包含全部20个 samples）保留在本地唯一命名 `/tmp` 日志，没有覆盖旧 agent artifact：

- 首轮 no-flush：`/tmp/qsa_stage3_noflush_W0XZ9d.log`
- cold：`/tmp/qsa_stage3_cold_xFGPU4.log`
- 倒序 no-flush 复测：`/tmp/qsa_stage3_noflush_repeat_vpZrVG.log`
- Stage1 回归：`/tmp/qsa_stage3_dense_regression_emIImk.log`

**异常如实保留**：首轮 no-flush 1K prefill pipeline scalar/WG8/WG16 为
0.705806 / 0.192306 / 0.214212ms；同轮 finalizer-only 为0.419828 / 0.111817 / 0.124859ms。
比后续 cold 和倒序复测绝对时间明显更大。未采集实时频率/功耗，不能断言具体原因（例如 DVFS）。
即使100次迭代 warmup，也不足以保证所有短 workload 的跨进程绝对时间稳定。
下表采用**完整倒序复测**作为 no-flush 主表，不逐项挑最小值、不删除异常 samples、不与 cold 混算倍率。
两轮 no-flush 的2K/4K prefill及全部decode趋势一致；首轮1K倍率方向也一致。

## S3.5 完整 score→topk：测得可选路径与 scalar fallback（ms）

选择列是前述人工策略，4K prefill 直接保留实测 scalar；不是声称实现了新的自动策略 dispatch。

### No-flush（倒序复测）

| 模式/size | 选择 | baseline GPU | 选择 score | 选择 topk | 选择 GPU total | 倍率 | baseline host → 选择 host |
|---|---|---:|---:|---:|---:|---:|---|
| prefill1024 | WG8 | 0.109499 | 0.000947 | 0.028906 | 0.029853 | 3.668x | 0.129450 → 0.051884 |
| prefill2048 | WG8 | 0.153661 | 0.001427 | 0.064781 | 0.066207 | 2.321x | 0.173384 → 0.087604 |
| prefill4096 | **scalar fallback** | 1.776103 | 0.061536 | 1.714567 | 1.776103 | 1.000x | 1.795893 → 1.795893 |
| decode4096 | WG16/P16 | 0.715332 | 0.007489 | 0.060947 | 0.068437 | 10.452x | 0.735295 → 0.088855 |
| decode16384 | WG16/P16 | 2.134036 | 0.008046 | 0.132047 | 0.140093 | 15.233x | 2.154819 → 0.159686 |
| decode65536 | WG16/P16 | 7.919400 | 0.017203 | 0.496723 | 0.513926 | 15.410x | 7.940162 → 0.533585 |

### Cold96MiB

| 模式/size | 选择 | baseline GPU | 选择 score | 选择 topk | 选择 GPU total | 倍率 | baseline host → 选择 host |
|---|---|---:|---:|---:|---:|---:|---|
| prefill1024 | WG8 | 0.110353 | 0.001770 | 0.029031 | 0.030801 | 3.583x | 0.130171 → 0.052460 |
| prefill2048 | WG8 | 0.154629 | 0.002400 | 0.064843 | 0.067244 | 2.300x | 0.174970 → 0.087931 |
| prefill4096 | **scalar fallback** | 1.780874 | 0.065416 | 1.715458 | 1.780874 | 1.000x | 1.800664 → 1.800664 |
| decode4096 | WG16/P16 | 0.829613 | 0.012328 | 0.061020 | 0.073348 | 11.311x | 0.850071 → 0.094109 |
| decode16384 | WG16/P16 | 2.243484 | 0.012864 | 0.132114 | 0.144978 | 15.475x | 2.263666 → 0.164775 |
| decode65536 | WG16/P16 | 8.017567 | 0.024786 | 0.496864 | 0.521650 | 15.370x | 8.034860 → 0.542092 |

GPU total median / P95（baseline → 选择）：

| 协议/模式 | size | median | P95 |
|---|---:|---|---|
| no-flush/prefill | 1024 | 0.109478 → 0.029843 | 0.109587 → 0.029999 |
| no-flush/prefill | 2048 | 0.153645 → 0.066249 | 0.153858 → 0.066353 |
| no-flush/prefill | 4096 | 1.775624 → 1.775624 | 1.780645 → 1.780645 |
| no-flush/decode | 4096 | 0.715311 → 0.068437 | 0.715624 → 0.068541 |
| no-flush/decode | 16384 | 2.134113 → 0.140103 | 2.134379 → 0.140208 |
| no-flush/decode | 65536 | 7.919427 → 0.513853 | 7.920103 → 0.514270 |
| cold/prefill | 1024 | 0.110311 → 0.030832 | 0.110420 → 0.031045 |
| cold/prefill | 2048 | 0.154583 → 0.067290 | 0.154895 → 0.067400 |
| cold/prefill | 4096 | 1.780103 → 1.780103 | 1.786618 → 1.786618 |
| cold/decode | 4096 | 0.829530 → 0.073332 | 0.830228 → 0.073546 |
| cold/decode | 16384 | 2.243488 → 0.144999 | 2.244390 → 0.145213 |
| cold/decode | 65536 | 8.017447 → 0.521666 | 8.018150 → 0.521979 |

## S3.6 消融：WG 大小、相同 scores 的 finalizer、partition 粒度

### 固定 score backend/partition512，完整 pipeline 的 WG 比较（mean ms）

| 模式/size | no-flush scalar | no-flush WG8 | no-flush WG16 | cold scalar | cold WG8 | cold WG16 |
|---|---:|---:|---:|---:|---:|---:|
| prefill1024 | 0.109499 | 0.029853 | 0.033265 | 0.110353 | 0.030801 | 0.034218 |
| prefill2048 | 0.153661 | 0.066207 | 0.072718 | 0.154629 | 0.067244 | 0.073786 |
| prefill4096 | **1.776103** | 1.848921 | 2.441848 | **1.780874** | 1.851702 | 2.447994 |
| decode4096 | 0.715332 | 0.297431 | 0.269166 | 0.829613 | 0.411802 | 0.383265 |
| decode16384 | 2.134036 | 0.456489 | 0.341322 | 2.243484 | 0.566052 | 0.450030 |
| decode65536 | 7.919400 | 1.164692 | 0.706140 | 8.017567 | 1.263936 | 0.805415 |

### Finalizer-only，同一 baseline GPU score buffer（mean ms）

| 模式/size | no-flush scalar | no-flush WG8 | no-flush WG16 | cold scalar | cold WG8 | cold WG16 |
|---|---:|---:|---:|---:|---:|---:|
| prefill1024 | 0.108577 | 0.028916 | 0.032348 | 0.109265 | 0.029562 | 0.033046 |
| prefill2048 | 0.152255 | 0.064817 | 0.071286 | 0.152869 | 0.065458 | 0.071911 |
| prefill4096 | **1.714958** | 1.786817 | 2.382286 | **1.722994** | 1.795958 | 2.392307 |
| decode4096 | 0.507843 | 0.089296 | 0.060994 | 0.513020 | 0.090776 | 0.062239 |
| decode16384 | 1.925307 | 0.248083 | 0.132036 | 1.940937 | 0.250557 | 0.133880 |
| decode65536 | 7.711708 | 0.955062 | 0.496692 | 7.769848 | 0.963598 | 0.500651 |

单 row 有可分摊的长扫描；大量短 row 已可由不同 scalar work-items 并发，增加每 row 的线程、
leader reduction 与 barrier 不一定合算。以上数据证实退化存在，但没有硬件 counter/GRF spill dump，
不把某一微架构瓶颈当作已经测实的唯一原因。

### Decode partition 消融（参数调优，不改 score 算法）

score 列取该 partition 的 WG16 pipeline 中 score event；scalar total 是保留 scalar finalizer 的消融项。

| past | partition | no-flush score | no-flush scalar total | no-flush WG16 total | cold score | cold scalar total | cold WG16 total |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4096 | 512 | 0.208156 | 0.715332 | 0.269166 | 0.322192 | 0.829613 | 0.383265 |
| 4096 | 16 | **0.007489** | 0.515724 | **0.068437** | **0.012328** | 0.520431 | **0.073348** |
| 4096 | 32 | 0.014026 | 0.521848 | 0.075030 | 0.021755 | 0.530098 | 0.082765 |
| 4096 | 64 | 0.026864 | 0.534640 | 0.087817 | 0.041546 | 0.549614 | 0.102614 |
| 16384 | 512 | 0.209177 | 2.134036 | 0.341322 | 0.317838 | 2.243484 | 0.450030 |
| 16384 | 16 | **0.008046** | 1.933030 | **0.140093** | **0.012864** | 1.938379 | **0.144978** |
| 16384 | 32 | 0.014833 | 1.939989 | 0.146874 | 0.022734 | 1.947556 | 0.154786 |
| 16384 | 64 | 0.026875 | 1.952562 | 0.158958 | 0.042364 | 1.968332 | 0.174457 |
| 65536 | 512 | 0.209567 | 7.919400 | 0.706140 | 0.308708 | 8.017567 | 0.805415 |
| 65536 | 16 | 0.017203 | 7.728395 | 0.513926 | 0.024786 | 7.737291 | 0.521650 |
| 65536 | 32 | 0.017500 | 7.728974 | 0.514228 | 0.024875 | 7.736859 | 0.521744 |
| 65536 | 64 | 0.030244 | 7.740854 | 0.526999 | 0.043005 | 7.753978 | 0.539869 |

首轮 no-flush 的64K WG16/P16与P32 total 分别0.513947/0.513765ms；倒序复测为0.513926/0.514228ms，
cold 为0.521650/0.521744ms。这不足以确认64K应该切到32，故不安装这个细粒度阈值。
即使 partition16 让 scoring 变快，仅靠参数更改仍留下 scalar topk 的主要成本；两项收益分开列出，
没有把全 pipeline 10–15x 都归给 cooperative scoring。

## S3.7 未改动证明与限制

三个新增源文件在完成 GPU 测试时的 SHA256：

| 文件 | SHA256 |
|---|---|
| `kernels/qsa_topk_cooperative.cm` | `66f6e3acc7c93ec3347066273a55dbefbd8e9404c398b8730f13d0bfaba1ff90` |
| `qsa_topk_cooperative.py` | `cb014c74eba9fa40e4c91f48836fd432ae559976e7f3103e3128ca092df392bf` |
| `validate_q2_cooperative.py` | `e07df70df277f0b880ccd3f744068de1d423edc8738aff4109cd534e8ef309ef` |

Stage2 的 prepare/project/helper/validator/driver 五项 SHA 与 S2.4 完全相同；两个 Q3 kernel、
`qsa_dpas_common.hpp` 与 Stage1 表相同。原 `validate_dense_bypass.py` SHA 仍为
`323439f2196eefc0f2b49cdde3567b9bb64997ca5a2982d47921920794ad1f93`。
所有旧 Q2 kernels/shared headers 都没有编辑或同步覆盖。

限制：只测 standalone synthetic Q2；没有 Q1→Q2→Q3/模型输出质量或完整 attention/model speedup；
没有 score 候选截断，但 uint32 索引/调用者 buffer 合同依然存在；没有 GRF spill、功耗或硬件访存计数测量；
只有一台 B580 与三轮有限样本 sweep，不能把时延精确到最后几位当作可跨机器重现的常数。
现有默认与 scalar fallback 都保留，其他 shape 需先测再决定，不继续 Stage4。

---

# Stage 4：Standalone decode split-selected-KV Q3

2026-09-10。本节追加前，Stage1/2/3 报告共 **47719 bytes**，SHA256 为
`d29a4a996f83bfcd4a8b5f5729eca977efd681fe21f9a725978c62fcb494c7ac`。
以上历史章节原样保留；先按原始字节长度检查 prefix，再检查本地/远端完整文件 hash。
**只新增五个代码文件、追加本报告；没有编辑任何既有 kernel、header、Python driver 或 Stage1/2/3 文件。
没有 Stage5，没有 OpenVINO/模型集成，没有偷偷更换 production/default dispatch。**

结论：在 B580、默认 Hq24/Hkv2/Dh=Dv256/r4/PA16/K512 的 B1/B4 decode、past2048/4096/16384 上，
实测 **P16** 是 P1/2/4/8/16 中最快的完整 Q3（包含 reducer）。首轮 no-flush 相对当前 scalar
分别约 **B1 11.64–11.67x、B4 4.07–4.40x**；cold 为 B1 12.19–12.32x、B4 4.30–4.39x。
反例也保留：短上下文 past1/3 的所有 split variants 都比 scalar 慢，**回退 scalar**。
这里只给人工显式 opt-in 建议，不安装自动阈值；不是整个 attention 或模型加速。

## S4.1 新文件、布局与算法

| 文件 | 用途 |
|---|---|
| `include/qsa_split_span.hpp` | 私有复制 scalar `qsa_softmax_step` / `qsa_attend_span`，保留 Intel Apache-2.0 许可；两函数签名/函数体与原文件逐字符校验 |
| `kernels/qsa_sparse_attention_split.cm` | 独立 producer，每个 head/token/partition 一个线程，写 FP32 acc 和 m/l |
| `kernels/qsa_sparse_attention_split_reduce.cm` | 独立稳定 softmax 合并，归一化并只应用一次 sigmoid 输出 gate |
| `qsa_attention_split.py` | `Q3Attention`，默认 `partitions=None` 调用未改动 scalar；显式 P 才调用两个 kernels |
| `validate_q3_split.py` | 严格 NumPy FP32 + 实际 scalar 对照、边界/guard/合并测试、交替完整 Q3 A/B |

没有从旧文件抽取公共 helper、没有改旧 include，也没有 `#define` 抑制旧 kernel 来复用源码。
新 header 的原始来源 SHA 是 `a731a8ac4367513101b13ec4a3b96a63d4f9898463dfc2328a75f36c45b15081`；
validator 同时检查 scalar 文件 SHA 和复制的两个函数的精确文本，防止未来无声漂移。

### Producer

- GWS=`[Hq,num_tokens,P]`，LWS=`[1,1,1]`，并显式检查 head/token/partition 越界。
- 前九个 tensor 参数沿用 scalar 顺序：Q、K、V、past、subsequence_begins、page IDs、page_begins、selected IDs、counts。
  output 替换成 FP32 partial_acc、partial_ml；随后六个 runtime scalars 原顺序：num_tokens、num_seqs、
  q_base、q_pitch、q_head_stride、gate_delta。producer 保留 gate_delta ABI，但不读取 gate。
- Q 的 half-element offset 仍是 `q_base + token*q_pitch + head*q_head_stride`；先转 FP32 再乘 scale。
  KV 仍为 `[physical_page,Hkv,PA,Dh/Dv]`，KV head=`head/(Hq/Hkv)`；selected metadata 每个 token 一份、各 head 共享。
- `selected=min(uint(selected_counts[token]),K)`；partition p 扫描连续 **slot** 区间
  `[floor(selected*p/P),floor(selected*(p+1)/P))`，不是切连续的绝对 KV 地址，也不是跨 query 求 selection union。
- `b<0` 在任何 `(uint)b*r` / page / KV 地址计算前 continue；没有先读 invalid 地址再 mask。
  有效 b 必须是本 query 可见的完整 block，正 ID/page 的合法性依然是 caller 合同。
- 仅最后一个 partition 处理 `[floor((abs_pos+1)/r)*r,abs_pos]` 的不完整 tail；最多 r-1 个 token。
  **K=0 仍执行真实 tail-only attention**，不是直接清零。tail 也为空时所有 partial l=0。
- 每个线程运行原 scalar online softmax，以 FP32 写未归一化 acc、running_max、running_sum；没有 half 中间值、没有 gate。

### Scratch 与 reducer

令 `row=(token*Hq+head)*P+p`：acc 为 `[tokens,Hq,P,Dv]`，ml 为 `[tokens,Hq,P,2]`，最后一维 `[m,l]`。
两块独立 FP32 连续 buffer，总容量 `tokens*Hq*P*(Dv+2)*4` bytes，**没有 K/V materialization**。
默认配置 P16：B1 **396288 B**、B4 **1585152 B**、B16 **6340608 B**，不含验证 guard。

reducer GWS=`[Hq,num_tokens,1]` / LWS=`[1,1,1]`，先只在 l>0 的 partition 上取全局 m，然后：

- `w_p = exp(m_p-m)`；`acc = sum(w_p*acc_p)`；`l = sum(w_p*l_p)`。
- **在读取 m、求 exp、加载 acc 前先跳过 l<=0**；empty m=-inf/NaN、empty acc=NaN 的合成测试也通过。
- 最后归一化并从原始 Q layout 的 `q_elem+gate_delta` 读取 gate，逐通道乘 sigmoid 一次，写 half output。
- 全 empty 时保持 acc=0、inv_l=0，输出精确零，绝不做 `-inf-(-inf)`。
- producer/reducer 连续 enqueue 于 clops 同一 **in-order queue**。没有 kernel 内跨线程 global memory 回读；
  out-of-order queue 需要调用方另外建立事件依赖。

### 显式 helper 合同

`Q3Attention(cfg)` 不改变任何原默认；`Q3Attention(cfg, partitions=16)` 才启用 split。
P1 也保留 producer+reducer 两次 dispatch，而不是自动偷换 scalar，便于测量真实开销。
接受整数 P=1..16；主要验证/计时 P1/2/4/8/16，额外实测 K7/P3/P15 的非整除情况。
`scratch_elements(num_tokens)` 返回两块 FP32 element counts；caller 分配 scratch/output 并维持生命周期。
enqueue 内无 device allocation、数据复制或 finish；空 token batch 是 host no-op。
测试 driver 对零长度 Q/K/V/page/selection/count 输入分配 dummy device buffer，K0 不上传零字节。

要求 Hq%Hkv=0、Dh/Dv 是正16倍数、PA%r=0、K>=0；Q base/pitches/gate delta 使用 nonnegative、dword-aligned half offsets。
caller 仍需保证所有 Q/K/V/output/selection buffer 足够大、metadata 匹配、地址和 selected*P 等 uint32 算术不溢出。
helper 检查 scratch byte-offset 上限，不是对所有大 shape 的完整地址安全证明。
**有效 counts 必须 0<=count<=K**；GPU 保留原 scalar 的 K 上界。额外 count=K+3 只是防越界鲁棒性测试，不扩张支持合同。
不支持任意非对齐 tensor layout，不增加 noncausal/有效 NaN 输入的语义，也不自动处理非法正 block ID。

## S4.2 严格正确性与原 Q3 回归

两次完整远端 edge suite 均通过，第二次增加 exact partition mass 和 P3/P15；所有测试失败均抛异常、非零退出。
第一次 CM 编译和数值测试即成功，**没有 CM 修复循环**；后续只加强新 validator，没有改变 kernel/helper/timed path。

28 个 cases，每 case 对 scalar + P1/2/4/8/16 都检查：

1. 独立 NumPy **FP32** stable softmax oracle：显式 FP32 Q/K/V、matrix dot、max-subtraction、exp、sum 和 PV，
   不复刻 GPU online recurrence 或 partition 算法。benchmark 也先做完整 oracle/scalar 对照，不盲计时。
2. mixed 37-token batch 包含 q_len0、past0/1/3/14/15/31/63、非零 global token begin、短 chunk、shuffled pages。
3. 真正 padded layout：q_base32、gate_delta=Dh+16、head_stride 额外 padding、token pitch 再加32；padding 置 NaN。
   计时配置使用紧密原始 `[Q|gate]`，q_base0。
4. B1/B4/B16 × past%4=0/1/2/3；r2/4/8/16 分别 PA8/16/32/64，并覆盖各 r 的 **全部** past remainders。
5. K0/K1/K4/K8/K512、稀疏/互不相交 selections、P>selected count、empty attention、tail-only、全零 query、ungated、Dh64/Dv128。
6. count 内有 -1 slots；count 外 padding 故意包含 INT_MAX，检测误扫描未选择 slots；额外 count>K 检查原上界。
7. output 和两块 scratch 尾部均有64元素 NaN guard；额外发出越界 head/token/partition work-items。
   所有有效输出/partial 必须有限；l>=0、l=0 的 acc 精确0；all-empty output 精确0。
8. Q=0 时每个 partial 的 l 必须精确等于该 slot 区间有效 blocks*r 加上仅最后 partition 的 tail 数，
   独立验证无重复/遗漏 tail、无错误区间划分。
9. 全空 batch、无 sequences 的 helper no-op，以及非法 P/geometry 的 host rejection。

另有 synthetic reducer 两种测试（m=1000/999、非均等 l=2/3、empty m=NaN/-inf、empty acc=NaN；以及全 empty），
验证稳定合并、零分母处理和一次 gate。输入对象明确持有至 finish。
新增 K7/P3/P15、第二随机种子29，分别与 FP32 和实际 scalar 比较；这两项不做速度推荐。

| 验证项 | 最大绝对误差 |
|---|---:|
| Stage4 edge suite，所有 split outputs vs NumPy FP32 | **0.000244140625** |
| Stage4 edge suite，所有 split outputs vs scalar | **0.000244140625** |
| 六个正式 benchmark shapes vs NumPy（两协议和复测一致） | **0.000014647841453552246** |
| 六个正式 benchmark shapes vs scalar | **0.0000152587890625** |
| K7/P3、P15 vs NumPy / scalar | 0.000115483999 / 0.00006103515625 |
| 原 `validate_q3_dpas.py` 18 cases vs 其 reference，worst | **0.000386536** |

Stage4 output 断言 `atol=2e-3,rtol=2e-3`，合成 reducer 为1e-3；guards/empty outputs/exact l 使用精确相等。
P1 在本轮所有普通 cases/benchmark 与 scalar 逐值差为0，但没有把它升级成所有输入的 bit-exact 合同。
P>1 的累加/归一化次序不同，少量 half 舍入差异正常；不承诺模型质量或端到端精度。
原18-case suite 的 DPAS baseline/optimized 仍逐元素完全相等，文件未修改。

## S4.3 实测协议与复现

远端 `openvino-ci-74@10.239.140.245`，Intel Arc B580、160 EUs，报告最大频率2900MHz；目录 `/mnt/river/qsa/cm_kernel`。
GPU 编译/测试全部在远端串行执行，本地只做 AST/Pylance syntax、editor diagnostics、日志统计和 hash。
所有 SSH 使用 BatchMode，无密码；外层 SSH 与远端 Python 都有300s timeout。包装：
`timeout 300 ssh -o BatchMode=yes openvino-ci-74@10.239.140.245 'cd /mnt/river/qsa/cm_kernel && env CM_FE_DIR=/mnt/river OPENBLAS_NUM_THREADS=1 timeout 300 .venv/bin/python <参数>'`。
现有 `cm_kernel/.env` 存在；未读取/修改/同步秘密。

| 实际远端参数 | 结果 |
|---|---|
| `validate_q3_split.py`，初版 | 28 cases + synthetic reducer 全通过 |
| `validate_q3_split.py --benchmark-only --warmup 100 --samples 20` | B1/B4 × past2048/4096/16384，36项 no-flush |
| 同上增加 `--flush` | 36项 cold96MiB |
| `validate_q3_dpas.py` | 未修改原18 cases 全通过 |
| `validate_q3_split.py --benchmark-only --sizes 1 3 --batches 1 4 --warmup 100 --samples 20` | 24项短上下文对照，scalar fallback |
| `validate_q3_split.py --benchmark-only --sizes 16384 4096 2048 --batches 4 1 --warmup 100 --samples 20` | 36项倒序 no-flush 复测，P16结论不变 |
| `validate_q3_split.py`，最终版 | 28 cases + exact partition mass + synthetic reducer + P3/P15 全通过 |

- 每个 shape 使用同一九个 device input buffers，variant 的 output/scratch 独立；scalar 始终是当前原 kernel。
- 每 variant 100 次 warmup、20 samples；偶数轮 scalar→P1→P2→P4→P8→P16，奇数轮严格反序。
- profiling 明确断言 scalar 每次1 event、split 每次2 events；GPU total 是两 event duration 相加，
  **不包括 queue gap/host launch**。另列 host enqueue→finish（包含 helper host checks、launch、queue gap、同步）。
- 无 CPU oracle/编译/分配/H2D/D2H 时间混入。cold 每完整 Q3 前独立 flush96MiB+finish，flush 不计时，
  **producer→reducer 之间不 flush**。warmup 不 flush。cold 是协议，不是硬件100% miss证明。
- 表格采用完整首轮 no-flush 与完整 cold，不挑最小 sample；倒序复测另列，不混算倍率。
- 原始 stdout 保存全部20个 GPU total 和 host samples、各阶段 mean、median/P95、scratch bytes 和 errors：
  `/tmp/qsa_stage4_noflush_2h2mod.log`、`/tmp/qsa_stage4_cold_ZP6GzH.log`、
  `/tmp/qsa_stage4_repeat_EkoeK4.log`、`/tmp/qsa_stage4_short_q04g4J.log`。
  验证日志：`/tmp/qsa_stage4_validate_uRFpiw.log`、`/tmp/qsa_stage4_final_validate_oAVNgO.log`、
  `/tmp/qsa_stage4_original_q3_5ihDB6.log`。这些是本地临时 artifacts，不保证永久保留。

## S4.4 完整 Q3 GPU total 消融（mean ms）

### No-flush

| B | past | scalar | P1 | P2 | P4 | P8 | P16 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2048 | 0.450693 | 0.457145 | 0.229197 | 0.125265 | 0.065004 | **0.038624** |
| 1 | 4096 | 0.451671 | 0.457494 | 0.230140 | 0.125457 | 0.065255 | **0.038703** |
| 1 | 16384 | 0.457932 | 0.463926 | 0.233328 | 0.127348 | 0.066306 | **0.039343** |
| 4 | 2048 | 0.486380 | 0.494937 | 0.249160 | 0.132353 | 0.149213 | **0.119416** |
| 4 | 4096 | 0.518567 | 0.524645 | 0.262432 | 0.138416 | 0.151905 | **0.121093** |
| 4 | 16384 | 0.627114 | 0.636035 | 0.315702 | 0.166421 | 0.177926 | **0.142437** |

### Cold96MiB

| B | past | scalar | P1 | P2 | P4 | P8 | P16 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2048 | 0.564776 | 0.571879 | 0.288582 | 0.147452 | 0.077161 | **0.045843** |
| 1 | 4096 | 0.571734 | 0.578994 | 0.291192 | 0.148686 | 0.078405 | **0.046806** |
| 1 | 16384 | 0.604494 | 0.611609 | 0.308614 | 0.156910 | 0.082884 | **0.049603** |
| 4 | 2048 | 0.564046 | 0.570770 | 0.289775 | 0.154942 | 0.160999 | **0.131281** |
| 4 | 4096 | 0.583151 | 0.590791 | 0.298833 | 0.160629 | 0.165233 | **0.133807** |
| 4 | 16384 | 0.655520 | 0.663322 | 0.338577 | 0.180900 | 0.184260 | **0.149239** |

P1 的实际 producer 工作和单-dispatch scalar 很接近，增加 scratch 写/读、reducer、第二次 launch 后更慢，
正式 shapes 两协议 GPU 退化约1.17–1.76%；因此不能把 P1 当作无成本 fallback。
B4 的 P8 比 P4 慢，分区数与速度**不单调**；没有硬件 counter 证据，不把某个缓存/调度瓶颈断言为唯一原因。

### P16 阶段成本与 host 时间（ms）

| 协议 | B/past | producer | reducer | total | scalar/total | scalar host | P16 host |
|---|---|---:|---:|---:|---:|---:|---:|
| no-flush | 1/2048 | 0.033494 | 0.005130 | 0.038624 | 11.669x | 0.465741 | 0.059009 |
| no-flush | 1/4096 | 0.033588 | 0.005114 | 0.038703 | 11.670x | 0.466533 | 0.059495 |
| no-flush | 1/16384 | 0.034208 | 0.005135 | 0.039343 | 11.639x | 0.471847 | 0.059956 |
| no-flush | 4/2048 | 0.112489 | 0.006927 | 0.119416 | 4.073x | 0.500676 | 0.140702 |
| no-flush | 4/4096 | 0.114328 | 0.006765 | 0.121093 | 4.282x | 0.532901 | 0.142081 |
| no-flush | 4/16384 | 0.135239 | 0.007197 | 0.142437 | 4.403x | 0.641715 | 0.162754 |
| cold | 1/2048 | 0.040442 | 0.005400 | 0.045843 | 12.320x | 0.581344 | 0.065516 |
| cold | 1/4096 | 0.041333 | 0.005474 | 0.046806 | 12.215x | 0.586468 | 0.067263 |
| cold | 1/16384 | 0.044172 | 0.005432 | 0.049603 | 12.187x | 0.620104 | 0.081597 |
| cold | 4/2048 | 0.124197 | 0.007083 | 0.131281 | 4.296x | 0.579799 | 0.153942 |
| cold | 4/4096 | 0.126854 | 0.006953 | 0.133807 | 4.358x | 0.598057 | 0.154847 |
| cold | 4/16384 | 0.142130 | 0.007109 | 0.149239 | 4.392x | 0.670335 | 0.170008 |

各列独立四舍五入，显示的 stage 之和可能与 total 最后一位不同。
GPU total median / P95（scalar → P16）：

| 协议 | B/past | median | P95 |
|---|---|---|---|
| no-flush | 1/2048 | 0.450729 → 0.038697 | 0.450833 → 0.039062 |
| no-flush | 1/4096 | 0.451666 → 0.038749 | 0.451770 → 0.039067 |
| no-flush | 1/16384 | 0.457916 → 0.039322 | 0.458125 → 0.039588 |
| no-flush | 4/2048 | 0.486250 → 0.119478 | 0.487104 → 0.120108 |
| no-flush | 4/4096 | 0.518281 → 0.121093 | 0.520641 → 0.121691 |
| no-flush | 4/16384 | 0.630781 → 0.142239 | 0.633749 → 0.143588 |
| cold | 1/2048 | 0.564583 → 0.045832 | 0.566057 → 0.046161 |
| cold | 1/4096 | 0.571666 → 0.046770 | 0.572400 → 0.047521 |
| cold | 1/16384 | 0.604583 → 0.049583 | 0.605525 → 0.050213 |
| cold | 4/2048 | 0.564218 → 0.131197 | 0.564718 → 0.132015 |
| cold | 4/4096 | 0.583437 → 0.133749 | 0.585213 → 0.134588 |
| cold | 4/16384 | 0.675104 → 0.152811 | 0.677942 → 0.153957 |

## S4.5 反例、复测与选择范围

### 短上下文：保留已验证 scalar fallback（no-flush mean ms）

past1 为两个可见 tail tokens、没有完整 block；past3 为一个完整 block、没有 tail。

| B/past | scalar | P1 | P2 | P4 | P8 | P16 | scalar host → P16 host |
|---|---:|---:|---:|---:|---:|---:|---|
| 1/1 | **0.008760** | 0.013874 | 0.014202 | 0.015124 | 0.018218 | 0.026249 | 0.026378 → 0.055327 |
| 1/3 | **0.002968** | 0.004453 | 0.004415 | 0.004650 | 0.005551 | 0.007910 | 0.016847 → 0.028182 |
| 4/1 | **0.001953** | 0.002989 | 0.002931 | 0.003124 | 0.004270 | 0.006343 | 0.015344 → 0.025568 |
| 4/3 | **0.002328** | 0.003358 | 0.003207 | 0.003484 | 0.004708 | 0.006796 | 0.015608 → 0.025816 |

首个短 shape B1/past1 的绝对时间明显高于后面的短 shapes；原样报告，没有归因于未采集的实时频率/功耗。
这组反例只支持保留 fallback，不能据此推导精确 crossover，也未做短 shape cold 计时。

### 倒序 no-flush 复测（完整 sweep，mean ms）

| B/past | scalar | P1 | P2 | P4 | P8 | P16 |
|---|---:|---:|---:|---:|---:|---:|
| 4/16384 | 0.564343 | 0.567932 | 0.284952 | 0.149973 | 0.155963 | **0.126707** |
| 4/4096 | 0.510937 | 0.517957 | 0.259650 | 0.138119 | 0.150484 | **0.121218** |
| 4/2048 | 0.500328 | 0.508885 | 0.256228 | 0.135744 | 0.151338 | **0.120176** |
| 1/16384 | 0.457260 | 0.463624 | 0.232640 | 0.127155 | 0.066119 | **0.039182** |
| 1/4096 | 0.450812 | 0.457723 | 0.229630 | 0.125385 | 0.065452 | **0.038676** |
| 1/2048 | 0.450734 | 0.457270 | 0.229583 | 0.125249 | 0.065108 | **0.038635** |

B4/past16K 首轮 scalar/P16 为0.627114/0.142437，复测0.564343/0.126707，约10–11%的绝对时延变化；
cold 同 shape median 也高于 mean。没有删除异常样本、没有挑更好一轮作主表；相对 P16 最优的结论在三轮一致。
100次 warmup 不保证跨进程/输入缓存工作集/设备状态的绝对时间恒定。

**人工 opt-in 选择**：当前 B580 默认配置、B1/B4、已测三个长 past、每 token selected=512 时，可显式 P16。
past1/3 保留 scalar；P1 不推荐作为性能路径。其他 batch/设备/布局/selection 分布、少量 selected、prefill 或不同 r/Dh/Dv，
默认 scalar/原 driver，不从本轮数据猜测全局路由。B16 和 r2/8/16 只有正确性验证，没有性能推荐。

## S4.6 限制、源文件校验与保留范围

- r16、Dh=Dv256 的编译器明确报告 **3136-byte spill**，原 scalar 和五个 split producers 都如此；
  数值通过，但不调整已验证 scalar helper 来隐藏该问题，也没有对此配置宣称加速。
  默认 r4 的这次编译没有 spill warning；未采集完整 IGC asm/硬件 cache counters，不能泛化成所有配置零 spill。
- 新 kernel 仍是每 head 独立 scalar dot/online softmax，不共享 GQA KV、不改 DPAS tile，不做 Stage5。
  大 P 增加并发也增加 query/scratch/reduction 开销；P16 只是在测过的五个候选中最好，不是全局最优证明。
- 正式测量是 standalone synthetic Q3，selected 输入由 harness 生成；没有 Q1→Q2→Q3、完整 attention、模型质量或模型时延测试。
  原 Q3 suite 重跑了；Stage1/2/3 本轮只做 source/report preservation，不声称重新执行它们的完整 GPU suites。
- K0 dummy buffers、empty-partition NaN tests 不等价于支持任意损坏 metadata；counts/有效ID/页表/地址范围合同不可省略。

最终五个代码文件 SHA256：

| 文件 | SHA256 |
|---|---|
| `include/qsa_split_span.hpp` | `9f34c000e4fb34e08783cb1a65f475446b8bd51443697c0a637d8fc255b0613e` |
| `kernels/qsa_sparse_attention_split.cm` | `918c448bccc5ff531baf498733f6869fc461e7ba5704955b4084e9eb905a4079` |
| `kernels/qsa_sparse_attention_split_reduce.cm` | `746db0864f07dc4a6b6974fe73394f309f36623e009ac4f1da4bba143b34a65e` |
| `qsa_attention_split.py` | `79305993fcc57ce945262809148336b24e53d84786371de25403d15f2b5c0836` |
| `validate_q3_split.py` | `6e9f0559b3fbd0a4396ed13c0734ac72635c90b2191b8db1823ce22b6ba1a559` |

只 rsync 明确枚举的上述五个路径，最终单独 rsync 本报告；没有覆盖整个目录、没有 `--delete`、没有同步父 config 或 `.env`。
两个旧 Q3 kernels 与 `qsa_dpas_common.hpp` SHA 仍与 Stage1 相同；原 Q3 validator SHA 仍为
`b31c487e90a4bd2110ca4377050e12bc442658220f135a014676a629e0b93e3e`。
Stage1/2/3 实现/验证源码保持原 hash；报告前47719 bytes 与原 SHA 相符；同步后本地/远端新源码和完整报告 hash 一致。

---

# Stage 5：Standalone 单 token × 全 GQA heads 的 Q3 DPAS tile

2026-09-10。本节追加前，Stage1–4 报告共 **68025 bytes**，SHA256
`5fa001e05720e361cca6c06cd752e51475ca51e1363351e486cd8c27810cc60a`。
以上原始字节保留。只新增三个文件、追加本报告；**没有编辑任何旧 kernel、header、helper、测试或默认分派**。
没有 OpenVINO/模型集成，没有继续改 Stage4，没有实现 split-selected-KV GQA。

**结论：新 kernel 真实远端编译成功，29 项 Stage5 数值/边界测试完成；原18项 Q3 回归全部通过。
默认模型几何下，对当前 optimized DPAS（不是旧 baseline）的串行100/20测量，所有本轮正式 shapes 均更快：
past0 independent q1K/2K/4K/8K no-flush 为1.168/1.154/1.286/1.765x；
past16K independent q128/512 为4.010/2.581x。仍为显式 opt-in，没有自动默认或虚构 crossover。**

## S5.1 文件与新 tile

| 新文件 | 用途 |
|---|---|
| `kernels/qsa_sparse_attention_gqa_dpas.cm` | 独立 CM kernel，单 token/all-GQA-head N16、W4、单 selected list |
| `qsa_attention_gqa.py` | `Q3GQAAttention`，默认 scalar，显式 enabled 才启用；可显式选择不支持几何的 scalar fallback |
| `validate_q3_gqa.py` | FP32/scalar/baseline-DPAS/optimized-DPAS 数值对照、边界测试、交替 benchmark |

新 CM 保留原 Intel Apache-2.0 许可，参考旧 DPAS 的 loaders、QK/PV 和在线 softmax，但不是包含旧 kernel
后修改宏来伪装新路径。旧 `qsa_sparse_attention_dpas.cm` 与 `qsa_dpas_common.hpp` 均一字未改。

- **WG = 一个 token × 一个 KV head**，local size `[4,1,1]`，global size `[Hkv*4,num_tokens,1]`。
  四个 workers 分别拥有 Dh/Dv 的四分之一；不再为一个 token 的12 heads发三个独立HT4 tile。
- QK 的 **N=16 是 heads**，M=8 是 KV rows，K=16 是 head dims；nrep12填前12列、后4列padding。
  支持1..16，不要求 nrep 整除16。PV 仍为 M8 heads × N16 V dims × K16 KV；P 经16×16 transpose后作为A。
- Q 的基址为 `q_base + token*q_pitch + kvhead*nrep*q_head_stride`。
  transposed d32 descriptor **rows=nrep，pitch=q_head_stride*2，width=Dh*2**，例如12行一次load填入16列，
  descriptor边界自动补零；每worker只读自身Dh slice。gate、head padding或相邻token不是Q维度的一部分。
  attention scale×log2(e)仍先以half折入Q，和旧DPAS一致；不是改变为scalar FP32 scale语义。
- KV 每worker/dimension slice只加载一次，供本token全部实际heads计算；这就是本阶段要求的
  **新 tile / 共享 KV**。没有额外跨head团队SLM KV materialization。QK的partial St通过双缓冲SLM求和：
  `[2][4][16][16] float`，共 **8192B**，每step一次 local fence + barrier。
  在线max/sum由每worker重复计算，PV和最终V slice保留在本worker，无第二次归约或全局scratch。
- **没有TT-way union、cursor数组或每token bitmap**。直接按单token有序selected list每16 KV tokens一组读取；
  page-local连续组coalesce，否则按r分块；所有2D message列宽≤32 half（64B）。物理页打乱仍走page table。
- `[floor((abs_pos+1)/r)*r,abs_pos]` tail最多r−1 tokens，独立最多一个step；该token/KV组只处理一次。
  partial pair由descriptor rows裁剪补零。K0仍算tail，tail也空时输出精确0。masked finite sentinel显式清零概率，
  不依赖 `exp(sentinel-sentinel)` 自动为0。
- epilogue按实际head行归一化，并用同一Q布局的gate_delta逐维sigmoid一次。
  **仅存h<nrep**，不写padding heads；输出连续 `[tokens,Hq,Dv]`，不把head列误当token。

### ABI / helper 支持合同

前九个tensor仍为Q/K/V/past/subsequence_begins/page IDs/page_begins/selected/counts；随后是
`token_map,output,num_entries,q_base,q_pitch,q_head_stride,gate_delta`。
token_map是caller提供的int32 `[num_tokens,2]`，每token唯一一行 `(seq,global_token)`；
可用现有 `H.make_score_tile_map(batch,tile=1)` 构建。**不能传旧16-token tile map**，不能重复/漏掉token或越界seq。
map占8 bytes/token；不分配KV或softmax全局scratch。enqueue内部无分配、copy、finish。

`Q3GQAAttention(cfg)` 默认 **enabled=False → 原scalar**；`enabled=True`才编译新路径。
不支持几何默认ValueError，只有调用者显式 `fallback="scalar"` 才回退。编译器失败不会被catch后静默回退。
这个默认只属于新增helper；已有prefill driver仍保持其optimized DPAS默认，Stage4 helper也不变。

- 新路径：正Hq/Hkv、Hq%Hkv=0、1≤nrep≤16；Dh/Dv在128/256中（W4 slice需32-half对齐；>256未验证故拒绝）。
  r∈{2,4,8,16}，page>0且page%r=0，K≥0，causal。实际pages8/16/32通过；不声称任意页大小都做过GPU验收。
- Q base、token pitch、head stride用half-element计数，要求32-byte对齐；gate_delta为dword对齐，
  head/token之间必须有足够空间。支持q_base/pitch/headstride/gatedelta padding，不支持任意字节strided布局。
  本轮padded gated默认：q_base32、gate_delta272、headstride544、pitch13088 half；ungated headstride272。
- Dh64等旧测试几何由显式fallback跑scalar；没有声称W4新kernel支持小于128的维度。
  noncausal包括fallback都拒绝。空batch为host no-op。
- counts必须在[0,K]，count内为升序、唯一、非负、causal完整block IDs；count外padding不读。
  **不支持Stage4的count内−1洞或overcount鲁棒性扩展**；新helper不读取GPU metadata验证这些条件。
  有效正ID/page/buffer容量、map一致性和uint32地址不溢出仍为caller合同；host仅额外检查Q/output地址上限。
  不支持有效NaN输入，不承诺无限shape范围或所有设备自动fallback。

## S5.2 严格验证、真实错误与原回归

所有检查失败均异常退出，`atol=2e-3,rtol=2e-3`，不在失败后放宽阈值。
FP32 oracle是显式float32 Q/K/V、完整stable softmax、矩阵PV，按GQA组向量化；不复刻DPAS半精度P或在线recurrence。
scalar、旧baseline和optimized均使用**相同九个device input buffers**；新旧map各自生成、输出独立。
旧baseline宏全0，optimized三个宏均1；两者输出仍精确相等。新GQA与optimized不要求bit-exact，因为KV分组/累加顺序变化。

29 cases：dense129 partial；53-token含空序列、past0/1/3/14/15/31/63、非零global begin的mixed batch；
qbase/token/head/gate padding；页交叉/连续和不连续blocks/partial steps；K0/K1/K7/K512；empty attention和tail-only；
gate off；Dh128/Dv256；全空和无sequence；independent/overlap/disjoint sparse；PA8/16/32；r2/8/16全部tail余数；
nrep1/3/6/12/16（DhDv128）；旧W8/HT4、W4/HT2、W4/HT1对照；Dh64旧W1/HT4和W1/HT8显式fallback；
第二seed29全零Q。正常随机seed17。每次验证额外dispatch越界KV-head/token或旧tile groups，output末尾64 half NaN guard保持不变；
所有实际输出有限，空attention精确0。完整输出对照也覆盖padding head误写后续组/token。
非法head/page/r/维度/noncausal和misaligned layout在host拒绝；不进行非法GPU地址访问来“验证”坏metadata。

| 项目 | 最大绝对误差 |
|---|---:|
| 新Stage5 edge GQA vs FP32 | **0.000376284122467041** |
| 新Stage5 edge GQA vs scalar | **0.00048828125** |
| 新Stage5 edge GQA vs optimized DPAS | **0.00048828125** |
| 全36条正式/复测benchmark GQA vs FP32 | **0.0003311634063720703** |
| 原未修改18-case Q3 suite vs原reference | **0.000386536** |

**不隐藏的失败记录：**

1. 新kernel首次CM编译即成功，前6 cases通过；首次在ungated padded case被helper过严的64-byte pitch检查拒绝。
   改为32-byte base/pitches后同case真实GPU通过。helper一次修正，新CM文件没有修复循环。
2. 新增Dh128/Dv256/W4对照触发 **旧DPAS** `IGC: Internal Compiler Error: Floating point exception`，原样重试复现；
   独立小case旧baseline成功、旧optimized失败、新GQA与scalar均成功。全suite中不为旧kernel改源码或隐藏异常：
   validator对这个具名额外shape显式输出 `UNSUPPORTED OLD DPAS COMPARISON`，仍严格验证GQA vs FP32/scalar。
   因此不能声称该shape的optimized对照通过；它不是原18项suite中的shape。
3. nrep16时旧heads_per_tile返回HT16/TT1，旧 `load_vnni_cols<1>` 编译报`cm_load`期望vector_ref<uint,16>、实参uint8。
   仅在新增validator选旧合法HT8/TT2作比较；新GQA仍一次处理全部16heads。没有修旧selector/shared header。
4. r16/DhDv256的原scalar编译报告 **3136B spill**（Stage4已有记录）；新GQA本轮没有spill warning。
   没有采集IGC全asm/硬件counter，不能把“没有warning”升级为全面零spill证明。

validator两次针对旧比较限制的修改，没有第三次失败修复；新CM从首次编译到最终测试hash保持不变。
Mem0服务不可达，使用已有repo memory和源码/报告证据；未读/同步任何秘密。

## S5.3 测量协议、复现及测量失误

唯一GPU：`openvino-ci-74@10.239.140.245` 的Intel Arc B580，160 EUs、报告最大频率2900MHz。
GPU仅在 `/mnt/river/qsa/cm_kernel`，`CM_FE_DIR=/mnt/river OPENBLAS_NUM_THREADS=1`、`.venv/bin/python`运行。
所有SSH为BatchMode；外层SSH与远端Python均timeout300。只rsync三个新增具名路径，最后单独同步报告。
`.env`已存在，仅检查存在性，未读取/修改/同步；没有同步父config、整个目录或`--delete`。

- 每shape先完整FP32/scalar/baseline-DPAS/optimized-DPAS/GQA输出检查；计时只保留**optimized DPAS vs GQA**。
  默认Hq24/Hkv2/DhDv256/r4/PA16/K512，旧W4/HT4，新W4/N16。
- 每variant100次warmup、20samples；每轮交替A→B/B→A，不是先独立跑完A再B；每次finish后读取**单GPU event**。
  GPU event不包括host launch/分配/编译/数据copy/FP32参考；另在原始JSON记录host enqueue→finish时间及全部samples。
  benchmark移除验证用额外越界WG。no-flush复用输入；cold每个variant前flush96MiB并finish，flush不计时。
  cold只是协议，不是硬件100% miss保证。
- past0的1K/2K是全选dense；4K/8K含dense前缀和sparse后缀。independent随机逐token选完整blocks；
  overlap选择共同最早K blocks；disjoint以token%4分成四个互不相交ID余数类。
  past16384/q128、512使用相同K512三种pattern。**全部是synthetic selections与随机Q/K/V，非模型trace，
  “真实配置”指模型几何而非真实模型产生的top-k**。本阶段没有宣称Q1→Q2→Q3/模型收益。

**测量操作失误明确记录：**最初误同时提交两条重复prefill benchmark，可能争用GPU。
`/tmp/qsa_stage5_noflush_Ur069M.log`、`/tmp/qsa_stage5_noflush_jYQPxS.log`整轮作废，未用于下表或阈值。
确认远端 `pgrep` 无validator进程后，重新**串行**执行以下全部正式测试；不通过挑选并发样本掩盖问题。

统一包装为 `timeout 300 ssh -o BatchMode=yes ... 'cd /mnt/river/qsa/cm_kernel && env CM_FE_DIR=/mnt/river OPENBLAS_NUM_THREADS=1 timeout 300 .venv/bin/python <参数>'`。

| 实际参数 | 有效结果/本地日志 |
|---|---|
| `validate_q3_gqa.py` 最终版 | 29项完成；`/tmp/qsa_stage5_validate_4723F2.log` |
| `validate_q3_gqa.py --benchmark-only --warmup 100 --samples 20` | 8项；`/tmp/qsa_stage5_serial_noflush_3oC7X5.log` |
| 上项增加 `--flush` | 8项；`/tmp/qsa_stage5_serial_cold_9duAZy.log` |
| `--benchmark-only --pasts 16384 --sizes 128 512 --patterns independent overlap disjoint --warmup 100 --samples 20` | 6项；`/tmp/qsa_stage5_serial_sparse_5guCsu.log` |
| 上项增加 `--flush` | 6项；`/tmp/qsa_stage5_serial_sparsecold_irlp7e.log` |
| `--benchmark-only --sizes 8192 4096 2048 1024 --patterns overlap independent --warmup 100 --samples 20` | 倒序8项；`/tmp/qsa_stage5_serial_repeat_7lMQr3.log` |
| 原 `validate_q3_dpas.py` | **18/18通过**；`/tmp/qsa_stage5_original_q3_co6JoL.log` |

错误日志：`/tmp/qsa_stage5_validate_usVtkE.log`（host alignment）；`ITFkG0`、`vZUR18`（同prefix/suffix，旧IGC错误）；
`/tmp/qsa_stage5_validate_DQjMUV.log`（旧HT16/TT1编译错误）。这些是本地临时artifacts，不保证永久保存。
本地两新Python文件Pylance syntax均无错误，editor diagnostics无错误；没有在本地GPU编译或运行。

## S5.4 正式结果（ms，O=当前optimized DPAS，G=新GQA）

表中每对数值均为 **O → G**；倍率=O mean/G mean，不与旧stage数字混算。

### past0：真实模型几何，query1K/2K/4K/8K

| 协议 | query/pattern | mean O→G | median O→G | P95 O→G | 倍率 |
|---|---|---|---|---|---:|
| no-flush | 1024 independent | 0.878833→0.752145 | 0.879010→0.752447 | 0.883489→0.755323 | 1.168x |
| no-flush | 1024 overlap | 1.091895→0.947323 | 1.201093→1.027969 | 1.207021→1.033203 | 1.153x |
| no-flush | 2048 independent | 3.163447→2.740974 | 3.164531→2.740677 | 3.172047→2.747745 | 1.154x |
| no-flush | 2048 overlap | 3.158531→2.742677 | 3.157916→2.741979 | 3.167416→2.751275 | 1.152x |
| no-flush | 4096 independent | 12.267104→9.542557 | 12.266510→9.542187 | 12.301494→9.555374 | 1.286x |
| no-flush | 4096 overlap | 8.966734→7.876984 | 8.969114→7.876458 | 8.987317→7.896807 | 1.138x |
| no-flush | 8192 independent | 41.694765→23.621302 | 41.704687→23.613228 | 41.779057→23.680953 | 1.765x |
| no-flush | 8192 overlap | 20.582755→18.243010 | 20.581510→18.246354 | 20.611921→18.274228 | 1.128x |
| cold | 1024 independent | 0.884213→0.753864 | 0.883125→0.752864 | 0.893463→0.761687 | 1.173x |
| cold | 1024 overlap | 1.009348→0.854979 | 0.886978→0.758228 | 1.267405→1.087176 | 1.181x |
| cold | 2048 independent | 3.156869→2.746682 | 3.154114→2.745937 | 3.182297→2.761671 | 1.149x |
| cold | 2048 overlap | 3.158296→2.747651 | 3.157864→2.745833 | 3.173650→2.760270 | 1.149x |
| cold | 4096 independent | 12.277135→9.561974 | 12.272396→9.563750 | 12.318557→9.574984 | 1.284x |
| cold | 4096 overlap | 8.952218→7.911229 | 8.954687→7.912083 | 8.974697→7.932698 | 1.132x |
| cold | 8192 independent | 41.700937→23.692333 | 41.722031→23.690312 | 41.798067→23.748052 | 1.760x |
| cold | 8192 overlap | 20.592588→18.336520 | 20.587343→18.337083 | 20.626968→18.379010 | 1.123x |

### past16384：synthetic sparse，K512

| 协议 | query/pattern | mean O→G | median O→G | P95 O→G | 倍率 |
|---|---|---|---|---|---:|
| no-flush | 128 independent | 2.344760→0.584796 | 2.344479→0.584843 | 2.366994→0.585442 | 4.010x |
| no-flush | 128 overlap | 0.826661→0.571192 | 0.826406→0.571198 | 0.832276→0.574275 | 1.447x |
| no-flush | 128 disjoint | 2.171145→0.828604 | 2.172187→0.831145 | 2.185724→0.839614 | 2.620x |
| no-flush | 512 independent | 5.318109→2.060088 | 5.318957→2.060156 | 5.331588→2.062187 | 2.581x |
| no-flush | 512 overlap | 1.520338→1.401244 | 1.520208→1.400624 | 1.524270→1.405833 | 1.085x |
| no-flush | 512 disjoint | 5.812151→2.981406 | 5.812604→2.977708 | 5.831260→3.000109 | 1.949x |
| cold | 128 independent | 2.361338→0.594458 | 2.358958→0.594583 | 2.377010→0.595016 | 3.972x |
| cold | 128 overlap | 0.702932→0.515661 | 0.702187→0.516562 | 0.709109→0.522838 | 1.363x |
| cold | 128 disjoint | 2.446177→1.056682 | 2.445000→1.057500 | 2.457656→1.062416 | 2.315x |
| cold | 512 independent | 5.335921→2.068635 | 5.336823→2.068750 | 5.344249→2.069916 | 2.579x |
| cold | 512 overlap | 1.710432→1.583323 | 1.709687→1.583958 | 1.717500→1.595052 | 1.080x |
| cold | 512 disjoint | 6.180359→3.282005 | 6.181458→3.285624 | 6.199614→3.290250 | 1.883x |

### 独立倒序no-flush复测（不替换主表）

| query/pattern | mean O→G | median O→G | P95 O→G | 倍率 |
|---|---|---|---|---:|
| 8192 overlap | 20.577583→18.245385 | 20.577604→18.240781 | 20.610068→18.281099 | 1.128x |
| 8192 independent | 41.725828→23.632593 | 41.720781→23.629374 | 41.846083→23.680911 | 1.766x |
| 4096 overlap | 8.961692→7.882270 | 8.962031→7.883593 | 8.987182→7.895380 | 1.137x |
| 4096 independent | 12.266838→9.538974 | 12.268906→9.541614 | 12.293932→9.553130 | 1.286x |
| 2048 overlap | 3.162890→2.743416 | 3.162708→2.743541 | 3.170895→2.750754 | 1.153x |
| 2048 independent | 3.160364→2.743213 | 3.159166→2.741041 | 3.168448→2.755672 | 1.152x |
| 1024 overlap | 0.879937→0.751885 | 0.880677→0.751510 | 0.883338→0.755312 | 1.170x |
| 1024 independent | 1.070088→0.903322 | 1.204322→0.901562 | 1.232744→1.059619 | 1.185x |

1K dense的independent/overlap实际都选全部可见完整blocks，却出现绝对时间差和明显mean/median偏离；
倒序后较慢项换成independent。保留全部samples和主表，没有归因于未采集的频率/功耗，也不把pattern标签当作dense差异的原因。
100次短warmup不能保证跨shape/进程设备状态一致。sparse overlap q128也出现cold比no-flush更快的绝对时间反转；
只能使用同轮交替比值，不将cold/no-flush差值直接解释为cache收益。

## S5.5 默认、人工分派建议与限制

**本轮默认模型几何的正式dense/sparse measurements没有观察到GQA回退**；不伪造dense退化。
但不因此宣称所有nrep、小query、r、Dh/Dv或设备都会快。padding列比例大、WG数量增加时可能不合算，尚未测量。

| 已测范围 | 人工建议（未安装auto dispatcher） |
|---|---|
| B580，Hq24/Hkv2/DhDv256/r4/PA16/K512，past0 q1024/2048/4096/8192，independent或overlap | 可显式 `enabled=True`；两协议和倒序复测均更快 |
| 相同几何，past16384、q128/512，三种synthetic sparse分布 | 可显式启用；independent收益尤其明显，不能把4x泛化为所有pattern |
| nrep1/3/6/16、DhDv128、r2/8/16、非默认pages/pitches | 本轮只有正确性，继续opt-in实验；没有性能阈值依据 |
| Dh64、nrep>16、其他不支持几何 | 明确ValueError，或caller显式scalar fallback；不要改变原driver |
| decode B1/B4、少selected、未测query长度/设备/真实model traces | 保持既有分派；Stage4显式P16/scalar建议不变，本轮未做新的decode A/B |

**没有测出精确连续dispatcher crossover**；不能从q128已快推导“q≥128都启用”，也不能从本组nrep12推导某个head阈值。
dense判定仍是每token `(abs_pos+1)//r<=K`，不是固定2048常量；新Q3本身不依赖dense分类。
若以后给自动路由，需要针对目标设备/实际selected overlap和短query重新验证，再处理混合sequence batch。

限制：仅standalone Q3，不含Q1/Q2、o_proj、模型质量或端到端速度；无真实模型trace。
不修改其他仓库；没有新的跨head SLM团队、prefetch、split KV或Stage4 reducer；没有性能计数器来确认单一瓶颈原因。
本次source verification和原18-case suite不等于重新执行Stage1–4全部GPU suite。

## S5.6 完整性

| 新源码 | SHA256 |
|---|---|
| `kernels/qsa_sparse_attention_gqa_dpas.cm` | `39a27c2885fef64fbe7e8accd33614b50d0d580c45c66fdad83f0d7376ebe36c` |
| `qsa_attention_gqa.py` | `6fbb54ba48541d9f9e4ea0402d05f8fbe8e509dee6ded6c9a030292f521df3f8` |
| `validate_q3_gqa.py` | `f849b4307262649401a41f37010e031a41a1b0910ac257baa79386f664aec9a8` |

修改前已列出所有46个既有 `kernels/*.cm`、`include/*.hpp`、顶层`*.py` hash；既有文件未编辑或同步覆盖。
最终上述49个本地/远端源码hash逐项一致，旧scalar/DPAS/公共DPAS头和原validator仍与S4.6历史hash一致。
报告append后按**前68025 bytes**验证原SHA，避免换行影响的错误prefix比较；报告单独同步后再比较完整hash。

---

# Integration：真实 Q0→Q1→Q2→Q3 standalone slice

2026-09-10，五阶段之后的最终集成。新增 **`test_qsa_pipeline.py`**，更新 `README.md`，
本报告只改总标题并追加本节。**没有修改任何已验证 kernel/header/helper/原 validator，也没有修改其他仓库或插件。**
追加前报告87831 bytes，SHA256=`1d0b395bff119865166b4f4fde69e0fdf76a954f31620996506e0c01495a193c`；
去掉首行后的原正文SHA256=`cdaa0785a03f7593e4942cedba1d7479e3af64d6c4ea804f74159852b1874fc4`。
本节不覆盖此前各阶段的历史限制；下面是新完成的 standalone 集成证据，不是模型部署结果。

## I.1 真正串联的路径与状态

新 driver 提供 `Fixture`、`State`、`Pipeline`；所有 device allocations/maps 在构造时完成，
`enqueue()` 在 clops **同一 in-order queue** 上依次运行完整 Q0/Q1/Q2/Q3，无阶段间 copy/finish/flush。
源码依据：`qsa_prepare_dpas.py`、`qsa_topk_cooperative.py`、`qsa_attention_split.py`、
`qsa_attention_gqa.py` 及对应 `kernels/`，旧 Q3 的三个优化宏显式均为1。

| 阶段 | baseline（默认） | optimized（明确 opt-in） |
|---|---|---|
| Q0 | existing paged K/V update | 相同 |
| Q1 | scalar prepare | `PrepareDPAS` 的 projection + FP32 prepare |
| Q2 prefill | original DPAS scorer + scalar finalizer，dense flags0 | coordinated dense bypass；全batch每row均dense才用WG8，否则scalar |
| Q2 decode | partition512 + scalar finalizer，dense flags0 | partition16 + cooperative WG16，dense flags1 |
| Q3 prefill | **CURRENT optimized DPAS W4/HT4**，不是历史未优化Q3 | 单token/all-GQA-heads W4 |
| Q3 decode | scalar | K512/B1或B4/每序列1query/all past≥2048才split-P16，否则scalar |

`--optimized` 只运行新路径，`--compare` A/B；不加两者只运行baseline。
这是本driver内部**显式选择后**的保守策略，不是插件默认自动部署；没有推导普适crossover。
配置限制为D2560/Hq24/Hkv2/Dh=Dv256/Hidx4/Di128/rot32/r4/PA16/gated causal；K0/K1仅安全测试。
原helper默认不变：Q1原driver scalar，TopKFinalizer scalar，PartitionScorer P512且score bypass off，
Q3Attention scalar，Q3GQAAttention enabled=False/scalar；没有传播到OpenVINO或其他repo。

**数据来源不是独立的 synthetic selection：**外部 mixed/Q/K/V/weights 是可重复合成输入；Q0写入实际KV，
Q1产生实际iq/raw/summary，Q2只读这些实际输出，Q3直接读Q2实际IDs/counts。未调用 `make_selection` 或随机summary生成器。
prefill past0从空cache开始。decode历史只在setup中用 **actual scalar Q0/Q1** 跑一次，并完整检查NumPy reference，
再把相同有效历史克隆成两个独立可变state；不是随机raw/summary假扮历史。history/setup不计入latency。
固定shuffle页池跨调用保留；新pipeline可接收前次同一device State与递增past。
重复enqueue固定past/current，确定性store；Q1仅重读past之前的旧raw，不会把当次half写出错误地用于fresh FP32 pooling。

## I.2 严格正确性：完整计数与near-tie诊断

最终全参考edge suite **9/9调用、2165 queries/路径**；正式5 shapes另有 **7173 queries/路径**。
两组不重复计数合计 **14个场景调用、9338 queries/路径**：两路径共 **18676行** 的IDs/counts/padding精确通过，
并且 **18676个query输出（每query24×256）全部通过各自NumPy attention reference**。
Q0整cache精确；Q1 iq/raw/summary与所有未触及slots、全valid scores、输出及scratch末端guard均检查。
正式no-flush先用108行/路径Q3抽样（包含全部changed rows）；随后cold `--full-ref` 对相同输入7173行全检查，
edge也追加 `--full-ref`，不是把抽样称作全参考。单独baseline/optimized两个CLI模式又分别跑完9项full-ref。

edge包含：17-token prefill；45-token mixed（empty seq、past3/15/2046/2050/2051/4096、页边界/partial tiles）；
4-token short decode；K0 mixed；K1 zero-mixed exact ties；全空batch；
2066-token双序列prefill `(0,2051),(0,15)` 后在**同一cache**实际续两次decode，分别到可见2052/2053和16/17。
显式断言可见2051/2052对应完整blocks512/513；旧raw参与完成summary，未完成summary保持原值。
所有cache/output/scratch额外64元素guard；验证时额外越界dispatch；K0有dummy backing，空batch不enqueue。
完整replay与100/20结束后再检查所有payload逐元素精确等于验证结果（包括NaN poison位置）。

数值断言未因集成差异放宽：Q1最终half用既有 `atol=8e-3,rtol=2e-3`；projection `3e-5/2e-5`；
Q2 valid scores对own iq/summary用 `3e-5/2e-5`；Q3 **`atol=rtol=2e-3`**，不是legacy宽松阈值。
最大绝对误差（完整edge+完整benchmark参考）：

| 比较 | baseline | optimized |
|---|---:|---:|
| Q1 iq vs FP32 reference cast half | 0.00390625 | 0.001953125 |
| Q1 raw / summary vs reference，各自最大 | 0.001953125 / 0.001953125 | 0.001953125 / 0.001953125 |
| Q1 DPAS projection vs FP32 BLAS | 不适用 | 4.2915344e-6 |
| Q2 scores vs own actual iq/summary NumPy | 4.2915344e-6 | 4.2915344e-6 |
| Q3 vs NumPy，**own actual selection** | **0.000321209431** | **0.000321209431** |

每row finalizer严格对照**自己的GPU scores**：score降序、同分ID升序取K，最终ID升序，padding全部−1；
dense/K0按确定性全选/空选处理，绝不读取poison作为reference。
优化dense整行维持NaN/−12345原始bits；每sparse有效score必须有限并符合reference。
已有独立Stage1/3动态read-count证据仍保留；本次没有新增硬件访存计数器或把poison测试冒充read-count插桩。

### A/B选择变化不是finalizer或attention失败

prefill1024/2048、B1/B4 decode以及全部edge的A/B selection完全一致。
**prefill4096仅3/4096行不同（3/2045 sparse rows）**，各交换一个block；两种cache协议完全复现。
这3行没有从attention reference中删掉，也没有对A/B硬套宽松输出容差。
每行记录全部score差的上界ε，并断言baseline kth-gap≤2ε；各自finalizer仍精确正确。

| token（0-based） | baseline移除 → optimized加入 | baseline kth-gap | optimized kth-gap | max score Δ（ε） | output max Δ |
|---:|---|---:|---:|---:|---:|
| 2735 | 161 → 379 | 0.000138163567 | 0.0000903606415 | 0.000239372253 | 0.001922607422 |
| 2791 | 594 → 248 | 0.0000163912773 | 0.00000119209290 | 0.000328302383 | 0.002023696899 |
| 3421 | 111 → 593 | 0.0000404119492 | 0.000312447548 | 0.000579357147 | 0.002233266830 |

该4K输入A/B iq/raw/summary各maxΔ=0.001953125，sparse score全局maxΔ=0.001597881317。
最终A/B maxΔ=**0.002233266830**，明确不是bit-exact质量承诺；其中超过0.002的行仍须通过自己的严格Q3 reference。
same-selection rows另做2e-3 A/B断言；1K/2K完整输出A/B逐值maxΔ=0，decode B1/B4为3.8146973e-6/1.5258789e-5。
这证明实现遵守各自选择，不证明真实模型质量不受near-tie变化影响。

首次新driver在empty-case出现NumPy空list默认float dtype导致boolean-mask `TypeError`；仅新Python修为显式bool。
之前已通过的5项和随后全部9项均重跑通过。**没有kernel修复，没有集成数值失败被放宽阈值遮盖**。

## I.3 完整slice实测（ms）

唯一远端B580，全部GPU命令串行；每shape/path **100 warmup + 20 alternating samples**，
偶数A→B、奇数B→A。baseline每sample5 events，optimized prefill6、split-decode7。
GPU列是完整event duration之和，不含event间gap；host列是enqueue→finish墙钟时间，包含launch/gap/sync，
不含编译、分配、H2D/D2H、history初始化或CPU reference。验证用额外WG不进入计时。
cold96MiB在**每完整slice之前**独立flush+finish，不计时，阶段之间不flush；不是100% cache miss证明。
没有从旧单stage表相加估计集成速度，没有挑最小值或剔除samples。

| 协议 | shape | baseline GPU mean | optimized GPU mean | 倍率 | baseline host mean | optimized host mean |
|---|---|---:|---:|---:|---:|---:|
| no-flush | prefill1024 | 6.454503 | 1.010185 | 6.389x | 6.492393 | 1.054527 |
| no-flush | prefill2048 | 13.692805 | 3.255930 | 4.205x | 13.731821 | 3.305981 |
| no-flush | prefill4096 | 33.572555 | 12.195325 | 2.753x | 33.613567 | 12.246815 |
| no-flush | decode past4096 B1 | 1.356638 | 0.120107 | 11.295x | 1.395673 | 0.170950 |
| no-flush | decode past4096 B4 | 1.929326 | 0.242184 | 7.966x | 1.969732 | 0.291798 |
| cold96MiB | prefill1024 | 6.450842 | 1.011930 | 6.375x | 6.487383 | 1.056469 |
| cold96MiB | prefill2048 | 13.685191 | 3.259826 | 4.198x | 13.715508 | 3.303229 |
| cold96MiB | prefill4096 | 33.573639 | 12.206820 | 2.750x | 33.611125 | 12.242902 |
| cold96MiB | decode past4096 B1 | 1.758931 | 0.135320 | 12.998x | 1.796153 | 0.186866 |
| cold96MiB | decode past4096 B4 | 1.939951 | 0.242591 | 7.997x | 1.978332 | 0.293070 |

GPU total median/P95（每对 baseline→optimized）：

| 协议/shape | median | P95 |
|---|---|---|
| no-flush/prefill1024 | 6.454218→1.009789 | 6.459164→1.013488 |
| no-flush/prefill2048 | 13.693592→3.255154 | 13.708253→3.267804 |
| no-flush/prefill4096 | 33.573331→12.193226 | 33.593644→12.217300 |
| no-flush/decode B1 | 1.356821→0.120103 | 1.361772→0.122184 |
| no-flush/decode B4 | 1.929842→0.242340 | 1.954123→0.245340 |
| cold/prefill1024 | 6.451091→1.010779 | 6.455118→1.016280 |
| cold/prefill2048 | 13.686196→3.260570 | 13.692503→3.267743 |
| cold/prefill4096 | 33.573071→12.204424 | 33.592248→12.226774 |
| cold/decode B1 | 1.758697→0.135309 | 1.760561→0.135632 |
| cold/decode B4 | 1.992706→0.250517 | 1.999471→0.251580 |

优化路径no-flush分stage mean，均来自同一次完整slice，不是独立synthetic子项：

| shape | Q0 | Q1 project | Q1 prepare | Q2 score | Q2 final | Q3 producer/单kernel | Q3 reduce |
|---|---:|---:|---:|---:|---:|---:|---:|
| prefill1024 | 0.008979 | 0.184224 | 0.031942 | 0.001318 | 0.029072 | 0.754651 | — |
| prefill2048 | 0.015442 | 0.364057 | 0.072265 | 0.002088 | 0.064916 | 2.737161 | — |
| prefill4096 | 0.029708 | 0.729588 | 0.161375 | 0.059619 | 1.710057 | 9.504979 | — |
| decode B1 | 0.000885 | 0.017385 | 0.001422 | 0.008234 | 0.053203 | 0.033755 | 0.005224 |
| decode B4 | 0.002005 | 0.022348 | 0.002583 | 0.012359 | 0.060473 | 0.134958 | 0.007458 |

4K prefill确实保留scalar finalizer；1K/2K只用WG8全选写出；decode用WG16/P16score与P16Q3。
cold B4的mean低于median，原始20项全部保留；没有实时频率/功耗telemetry，不能归因于特定GPU状态。
这里只报告两轮完整集成sweep的同轮倍率；既有Stage4/5的不同数据/独立cache条件不能混算。

## I.4 执行记录、复现与限制

完整命令已写入 `README.md`。每个远端Python命令均使用：

```bash
timeout 300 ssh -o BatchMode=yes openvino-ci-74@10.239.140.245 'cd /mnt/river/qsa/cm_kernel && env CM_FE_DIR=/mnt/river OPENBLAS_NUM_THREADS=1 timeout 300 .venv/bin/python test_qsa_pipeline.py <下面参数>'
```

| 实际参数 | 结果 / 本地原始stdout日志 |
|---|---|
| `--compare`（empty mask修正后） | 9/9；`/tmp/qsa_integrated_validate_final_JfKaO1.log` |
| `--compare --full-ref` | 9/9、2165全参考；`/tmp/qsa_integrated_edge_fullref_EwdMmN.log` |
| `--compare --benchmark-only --warmup 100 --samples 20` | 5 shapes、10 path rows；`/tmp/qsa_integrated_noflush_9KwOMQ.log` |
| `--compare --benchmark-only --flush --full-ref --warmup 100 --samples 20` | 5 shapes、7173全参考、10 path rows；`/tmp/qsa_integrated_cold_fullref_nbeZRT.log` |
| `--full-ref`，然后 `--optimized --full-ref`（串行） | 各9/9；`/tmp/qsa_integrated_modes_6VwVaI.log` |

首次host-only empty-mask错误保留在 `/tmp/qsa_integrated_validate_lpULVZ.log`，不用于性能证据。
所有BENCH JSON包含20个GPU/host原始samples、mean/median/P95、每kernel均值和scratch字节；临时日志不保证永久保存。
同步只枚举 `test_qsa_pipeline.py README.md PIPELINE_OPTIMIZATION_RESULTS_CN.md`，未同步整个目录、`.env`、父config或其他repo。
既有`.env`存在，未读取或修改秘密。Python AST/Pylance syntax/editor diagnostics检查通过；本地未执行GPU。

**范围限制：**这是完整standalone **QSA slice**，不是完整attention层或模型：不含外部Q/K/V投影、GR、o_proj、
page扩容/调度、其他网络层或真实模型trace/质量评测。CLI不等于生产API/插件部署；没有自动传播helper配置。
全token score scratch仍是 `[tokens,padded_max_blocks]`，1K/2K/4K分别1/4/16MiB；**尚未chunked**。
DPAS projection为2.5/5/10MiB；split Q3 B1/B4为396288/1585152B（不含guard）。
baseline为方便统一检查也分配未使用projection占位；history setup也非内存优化版本，都不计时，不宣称最小峰值内存。
新driver没有改任何旧kernel，因此未为本次“无kernel修改”重复整套Stage1–5独立suites；
其历史证据原样保留，本次验收依据是实际新链路全参考与exact metadata，而不是只比source hash。