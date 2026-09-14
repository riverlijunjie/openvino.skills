# PTL 长上下文 Q3 / Q2-topk 优化（2026-09-11）

## 范围与环境

- Standalone `cm_kernel`，远程 PTL Windows / Intel Arc B390，96 EUs、2400MHz、128KiB SLM。
- 参考硬件参数仍为用户提供的 FP16 58 TOPS、110 GB/s；没有套用 B580 参数。
- 几何不变：Hq24/Hkv2、Dh=Dv256、r4、selected512 compressed blocks、budget2048、PA16、causal、sigmoid gate。
- 真实测试流水线为随机但真实外部 x/Q/K/V → Q0 → DPAS Q1 → Q2 score/top-k → Q3；不是模型精度/任务质量测评。
- 本次没有修改或部署 OpenVINO 插件，也没有自动更改其它设备的 dispatch 默认值。

## 1. 原因：必须区分当前 GQA 内核与旧 union 内核

### Q3：主要是稀疏 KV 工作集/复用，而非跨 token union

当前 `qsa_sparse_attention_gqa_dpas.cm` 一个 WG 处理一个 token、一个 KV head，共享其 12 个 query heads。**没有跨 token selected-block union**，不能用旧 HT4/TT4 的 union 放大解释当前退化。

K 达到512后，每个 token 固定选择2048个完整 KV token，加0..3个尾部 token。忽略边界和尾部：

- 每个 token 的 K/V 请求量：2 KV heads ×2048×(256+256)×2 bytes = **4MiB**。
- 64K/128K 的整个 K/V cache 分别约 **128MiB / 256MiB**；其中每个 KV head 占64MiB /128MiB。
- 当序列增长而 top-k 固定，选中块在全历史中更稀疏，连续 PA16 合并加载更少，缓存复用更困难。
- 原 `(kv_head, token)` 工作组映射交错处理两个 KV heads；改为 `(token, kv_head)` 可以把一段调度集中到同一 KV head，缩小活跃工作集，增加相邻 WG 的复用机会。
- 原每个完整离散 step 4个 selected ID、4个 page ID 分别标量取数，存在连续的地址依赖；连续合并分支仅需1个 page ID。masked vector gather 可以成批取数。

**证据边界：**head-major A/B 显著改善，支持局部性/工作集解释；未采集硬件 L2 miss、TLB miss、stall counters，不能把每一项缓存贡献定量归因。上述请求字节不是 DRAM 实测字节，跨 WG 缓存命中会减少实际 DRAM 流量。

### Q2-topk：输入候选数增长 + 多次重复扫描

每行最多候选数为 `length/r`：64K为16384，128K为32768。

- Prefill 所有行的候选总数近似 `Q²/(2r)`，长度翻倍带来约4倍候选扫描工作，并非固定 K 就固定 top-k 成本。
- fast-WG16 的精确阈值选择每轮重新读取整行，各 worker 在寄存器中只保留 SIMD64 的当前片段；最后 greater/equal 统计、稳定 ID 输出又扫描两次。
- WG reduction 和整数比较也有成本；不能用 FP16 58 TOPS 当作 top-k 整数/同步吞吐。

## 2. 保留的优化

### Q3：head-major + vector metadata（W4、step16、GRF256 不变）

- `QSA_GQA_HEAD_MAJOR=1`：互换 NDRange 的 token/KV-head 轴，结果地址、算术顺序、KV/head 映射均不变。
- `QSA_GQA_VECTOR_META=1`：每step对 selected IDs 和 physical page IDs 分别 masked gather；未选中的 slot 不访问 metadata，尾部仅使用一个有效位置。
- API：`Q3GQAAttention(..., enabled=True, head_major=True, vector_meta=True)`。
- 两个开关默认 `False`，baseline 可直接复现，无近似、无额外 global scratch。

### Q2-topk：每 worker 缓存有序整数 keys

- 显式 finalizer：`fast-wg16-cached`。
- 容量：`min(2048, ceil(max_blocks/1024)*64)` keys/worker。
- 64K：1024keys/worker，4KiB寄存器数据；128K：2048keys/worker，8KiB。
- 每行加载一次 keys，阈值迭代、greater/equal统计和输出复用缓存。
- 容量不是截断上限：32769 candidates 需要2112keys/worker，超过2048时**整行一致回退全扫描**；保留精确 tie-break、signed-zero ordering、升序ID、chunk-local score addressing。
- dense/K0绕过仍然保留；没有增加 global scratch，所有构造/编译/分配均在计时外。

## 3. 测试方法与结果

完整流水线 cold96MiB 的主结果（ms）：

| 形状 | Q3 old→new | top-k old→new | 总延迟 old→new | 总加速 |
|---|---:|---:|---:|---:|
| prefill64K | 920.083→632.428 | 129.699→107.261 | 1126.354→813.736 | 1.384× |
| prefill128K | 2599.622→1794.950 | 436.186→349.859 | 3261.053→2361.888 | 1.381× |
| decode64K | 0.0825→0.0773 | 0.5575→0.0451 | 0.7373→0.2117 | 3.482× |
| decode128K | 0.0794→0.0770 | 1.0936→0.0794 | 1.3048→0.2874 | 4.541× |

Decode Q3未改，表中小幅变化是实测差异，不记作Q3优化收益。Prefill每种模式每个variant用20 warmup/20 samples；decode为100/20。

完整结果由 `analyze_ptl_optimization.py` 从带 `DONE`、`AB_PASS` 的完整日志生成：

- [分阶段/总延迟 A/B 表](PTL_Q3_TOPK_AB_RESULTS.md)
- 原始日志：`ptl_optimization_logs/`，结果表附完整日志 SHA256。
- `benchmark_ptl_optimized.py` 在**同一进程、同一设备状态、相同输入**上交替 baseline/optimized，分别测试 cold96MiB 与 no-flush。
- Prefill baseline 是原 fast-wg16 + GQA-W4，**不是更慢的128K legacy-scalar**。
- Decode baseline 是原 legacy cooperative-WG16 + split-P16；因此 decode 总收益同时包含 finalizer 算法选择和缓存，不应全部归功于寄存器缓存。
- Q3 只有 prefill 路径优化；decode 的 split-P16 + reducer 未改。
- 完整 top-k 每行使用自身 GPU scores 校验；Q1使用流式参考，Q3使用显式行采样 FP32参考、全输出 A/B比较、guards和确定性 replay。
- 所有投影、历史构造、NumPy参考、readback、cache flush和分配均不计入GPU kernel延迟。
- no-flush指交替variant共享缓存的协议，不是各自独立稳态；cold96MiB在整个流水线之前flush，不意味着经过Q0/Q1/Q2后Q3仍然完全冷缓存。
- 注意 iGPU 功率/频率和缓存影响会使未修改的 Q0/Q1/Q2-score 也出现小幅时间差；不能把总差异全部分解为严格隔离的单项收益。

### 验证精度与源版本说明

- Q3边界25 cases及host拒绝检查通过，baseline/head-major/head-major-meta均检查FP16 bits；覆盖r2/4/8/16、K0/K1/K7、空序列、尾部、异形Dh/Dv、padded heads、page8/16/32。
- Top-k通过52 finalizer configurations、4缓存fit/fallback边界配置、360 helper configurations（每项重放两次）和实际Q0/Q1/DPAS scores。
- Prefill64K/128K分别有104/297行FP32 Q3参考，所有63485/129021个稀疏行和2051个dense行均验证own-score选择；Q3最大参考绝对误差约0.00024134。
- 初次prefill性能日志的`full_payloads_bit_exact`标签实际使用NumPy数值精确比较（不区分±0/NaN payload）。已保留原测量源码`ptl_optimization_logs/benchmark_prefill_measured.py`，不重写历史日志。当前版本改为uint8逐字节比较，并单独执行`--validate-only`全尺寸复核；计时内核未变化。
- Decode性能日志使用增强后的uint8 A/B与全部post-timing payload检查。所有计时/复核版本的源码hash由ENV记录；kernel/source匹配不等于编译二进制hash验证。
- 独立64K/128K逐字节复核均通过，日志`ptl_bytecheck_prefill_65536.log`和`ptl_bytecheck_prefill_131072.log`均包含`VALIDATION_DONE`及`full_bytewise_ab=true`；这是独立正确性证据，不是新增计时样本。
- `--optimized --topk-finalizer fast-wg16-cached --score-row-cap 17 --full-ref`集成回归9次调用全部通过，见`ptl_optimization_logs/ptl_integrated_regression.log`。该回归使用原GQA调度；新head-major/vector-meta由上述全尺寸A/B和25项边界测试覆盖。
- 本地44项host/分析器回归通过；没有用CPU mock冒充GPU数值测试。

## 4. 消融：测试但没有保留的候选

使用256个末端长上下文查询进行同输入验证/测试（预取初筛32行）。这些是候选筛选，不是完整 prefill 性能。

| 候选 | PTL 结果 | 处理 |
|---|---|---|
| 下一步/下两步 KV prefetch | 32行128K冷缓存约0.991→1.132/1.154ms，变慢 | 删除；增加send和地址计算并未改善瓶颈 |
| demand-load cached/uncached/streaming hint | 256行128K约7.04–7.06ms，无稳定改善 | 删除 |
| W8/GRF256，W8/GRF128 | 比W4更慢，SLM交换/同步和工作线程增加 | 删除 |
| step32，摊薄softmax/barrier | W4基本持平，W8更慢 | 删除；寄存器/SLM负担抵消收益 |
| vector metadata单独启用 | 收益很小 | 与head-major组合保留 |
| head-major | 256行64K约5.36→4.03ms，128K约7.10→5.81ms | 保留 |
| head-major + vector metadata | 256行64K约3.98ms，128K约5.77ms | 保留；完整流水线及逐字节复核通过 |

旧候选日志保留用于审计；候选源码已删除，只保留当前胜出开关与baseline。

## 5. 复现入口

在远程目录 `D:\river\qsa\cm_kernel` 使用 `D:\river\py312\Scripts\python.exe`，设置 `OPENBLAS_NUM_THREADS=1`。

- GQA边界：`benchmark_ptl_q3_variants.py --edge-only`。
- GQA独立A/B：`benchmark_ptl_q3_variants.py --sizes 65536 131072 --rows 256 --warmup 20 --samples 20`。
- top-k精确回归：`validate_topk_fast.py`。
- 完整A/B：`benchmark_ptl_optimized.py --phase prefill --size 65536 --warmup 20 --samples 20`；替换phase/size测试其它形状。
- 标准单路径benchmark：`benchmark_long_context.py --phase prefill --size 131072 --topk-finalizer fast-wg16-cached --q3-head-major`。
- 分析：`analyze_ptl_optimization.py --check-sources`，额外校验实测kernel/wrapper与当前源码hash一致。
- Host测试：`python -m unittest test_topk_finalizer_options test_q3_gqa_options test_analyze_topk_pipeline test_analyze_ptl_optimization`。

## 6. 剩余限制

该优化改善长上下文性能，但不消除Q3每token稀疏KV请求和Q2 prefill近似二次候选数增长。后续应采集 cache/TLB/send/occupancy 硬件计数器，再考虑更细粒度的KV局部性调度或保持精确语义的分层选择。当前不建议凭FP16 roofline差距直接断言还有等比例可实现加速。
