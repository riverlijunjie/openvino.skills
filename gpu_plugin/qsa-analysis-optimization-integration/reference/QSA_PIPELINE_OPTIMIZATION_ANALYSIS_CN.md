# CM QSA 全流程性能分析与优化路线

## 1. 结论与证据范围

**可以继续明显减少现有实现中的冗余工作；优先级应从“只优化 Q3”转向 Q1 投影、Q2 选择以及 decode 并行度。实际加速倍率仍需逐项实现后测量。**

本报告基于：

- 算法背景：[QSA_GR_ANALYSIS_CN.md](../QSA_GR_ANALYSIS_CN.md)。
- 当前 `cm_kernel/` 源码，以及远程 Arc B580 上的补充阶段测试。
- 当前工作树中的 SGLang、llama.cpp、OpenVINO CM 调度代码；对应 HEAD 分别为 `3700c4ee26`、`050dde50c`、`2aaa541040`。HEAD 仅标识基线，不表示工作树没有本地修改。
- 上轮 Q3 优化及严格 A/B 结果：[Q3_DPAS_OPTIMIZATION_CN.md](Q3_DPAS_OPTIMIZATION_CN.md)。

本轮仅分析、运行已有测试并新增报告，**没有修改 kernel 或 OpenVINO 插件**。standalone Q3 的三项优化尚未同步到插件副本。

以下分别标注：**实测事实、源码事实、推导、待验证方案**。不把有用 FLOPs、逻辑字节数当作硬件计数器，也不将 NVIDIA 上的配置直接套用到 Intel GPU。

## 2. 算法中必须保持的语义

当前 `config.json` 经 standalone 配置解析后的主要参数：

| 参数 | 数值 |
|---|---:|
| 隐藏维度 D | 2560 |
| 主 attention Hq / Hkv | 24 / 2，GQA ratio = 12 |
| 主 attention Dh / Dv | 256 / 256 |
| Indexer Hidx / Di | 4 / 128 |
| 压缩率 r | 4 |
| token budget / block top-k | 2048 / 512 |
| PA page / 每页 summary 数 | 16 / 4 |
| standalone indexer rotary dimension | 32 |

Indexer Q/K 独立于主 attention Q/K。raw indexer K 先按 r 个 token 做 FP32 mean pooling，再做 RMSNorm、partial RoPE；summary 的 RoPE 位置是 block 起点。不能先逐 token norm/RoPE 再 pooling。

对绝对位置 t：

$$C_t=\lfloor(t+1)/r\rfloor,\qquad I_{t,b}=\frac{1}{\sqrt{D_i}}\sum_h\operatorname{ReLU}(q^{idx}_{t,h}\cdot\bar{k}^{idx}_b).$$

选择 `min(Kb, C_t)` 个完整 block，然后加上尾部 `[C_t*r, t]`。ReLU 必须在 indexer head 求和前；主 attention 则是普通 scaled-dot softmax，不能加入这个 ReLU。

- 同一 token 的选择集由所有主 attention heads 共享；不同 token 一般不共享选择集。
- 主 attention 使用原始主 K/V，不使用 summary 替代。
- 每个 query 最多访问 `2048 + 3 = 2051` 个 token。
- attention output gate、GR read gate、GR write gate 是不同运算；`o_proj` 及 GR 不属于当前 standalone Q3。
- CM Q3 的归并要求 selected block IDs 升序。改变 top-k 的输出顺序不是无关紧要的内部优化。

### 一个可精确利用的条件

当且仅当当前 query 的 `C_t <= Kb` 时，所有完整 block 都会入选，可以不算 indexer scores、不做 top-k，直接构造顺序 block IDs，或直接走等价的 causal dense attention。

对于本配置，该条件覆盖可见长度不超过 2051 的 query，而不是简单以 2048 为严格边界。**长上下文不能默认选择最早 512 个 block。**

即便跳过当前 query 的 selection，也必须维护 raw indexer K / summary 状态，供后续稀疏 query 使用。进一步可以只省掉这些 dense query 的 indexer Q 投影、Q norm/RoPE：它们不会被未来 query 使用，但 indexer K 投影不能省。

## 3. 新增远程阶段测量

设备：Arc B580，160 EUs；目录 `/mnt/river/qsa/cm_kernel`；解释器 `.venv/bin/python`；`CM_FE_DIR=/mnt/river`；`OPENBLAS_NUM_THREADS=1`。

本轮调用现有 `test_q0_kv_cache_update.py`、`test_q1_prepare.py`、`test_q2_score_tile_dpas.py`、`test_q2_score_partition.py`、`test_q2_topk_finalization.py`、两个 Q3 driver。统一 `--sizes 1024 2048 4096 --iters 10`，默认 PA16 / top-k512 / decode batch1；harness 默认 warmup10，每次计时前用 96 MiB buffer 冲刷缓存。数值是 GPU event 的 mean，单位 ms。

这是一轮定位瓶颈的测量，不是最终优化验收：输入是各 driver 独立生成的 synthetic tensors；top-k 输入是随机分数；Q3 输入是合成选择，不是本轮 Q2 的输出。下表不能作为集成流水线或完整模型的端到端测量。

### 3.1 Prefill（past=0）

| Query tokens | Q0 cache update | Q1 prepare | Q2 DPAS score | Q2 top-k | Q3 optimized DPAS |
|---:|---:|---:|---:|---:|---:|
| 1024 | 0.0106 | 5.2115 | 0.0113 | 0.3078 | 0.8833 |
| 2048 | 0.0187 | 9.6755 | 0.0245 | 0.7445 | 3.1583 |
| 4096 | 0.0374 | 18.9881 | 0.0674 | 2.1041 | 12.3013 |

直接结论：

1. **1K–4K prefill 当前最大阶段是 Q1，不是 Q3。**
2. Q2 的主要耗时在 top-k，不在已经 DPAS 化的 score。4K 时 top-k 约为 score 的 31 倍。
3. Q0 的量级很小，优先进一步微调 Q0 并不划算。
4. 1K/2K 的所有 query 都满足 dense 条件，却仍支付完整 score/top-k 开销。

若仅将独立阶段相加，Q1 在 1K/2K/4K 的占比分别约 81%/71%/57%。**这只是优先级模型**：实际 OpenVINO 中 Q0 与 Q1 无互相依赖，且缓存、CPU 调度、选择重合度均不同。

以 4K 的这个模型为例：如果 Q1 单阶段达到 2x，其余不变，则相加耗时约从 33.50 降到 24.00 ms，即 1.40x；若仅 Q3 达到 2x，则约 1.22x。这里只是 Amdahl 敏感性计算，**不是新优化的实测或承诺**。

### 3.2 Decode（每个请求只增加 1 token，batch=1）

| Past tokens | Q0 | Q1 prepare | Q2 partition score | Q2 top-k | Q3 scalar |
|---:|---:|---:|---:|---:|---:|
| 1024 | 0.0024 | 0.3654 | 0.1629 | 0.1820 | 0.2827 |
| 2048 | 0.0025 | 0.3638 | 0.3235 | 0.2582 | 0.5637 |
| 4096 | 0.0024 | 0.3557 | 0.3261 | 0.5221 | 0.5728 |

4K 时 Q2 score + top-k = 0.8482 ms，已经超过 Q3 的 0.5728 ms。因此 decode 不能只改 attention。

注意这些 past 都是 4 的倍数，此时新增 token 不完成新的压缩 block；Q1 decode 的完整评估还需覆盖 `past % r = 0,1,2,3`。当前 Q3 有用工作在预算饱和后基本固定，但 Q2 必须继续扫描历史 summary，不能据此宣称整个 QSA decode 对历史长度为常数复杂度。

### 3.3 与前次 Q3 验证的关系

上轮更严格的同输入交替顺序 A/B 使用 warmup100、2×30 次采样：top-k512 的 1K/2K/4K/8K 分别获得 1.214x / 1.204x / 1.144x / 1.127x。18 个回归用例通过，baseline/optimized 输出逐元素相同。

本轮 Q3 的 0.8833 / 3.1583 / 12.3013 ms 与上轮 0.8826 / 3.1577 / 12.3019 ms 接近。新的方案仍应沿用严格 A/B 与回归，而不是只依赖本轮十次采样。

## 4. 当前 CM 的主要结构性问题

### 4.1 Q1：“融合”了很多步骤，但投影没有使用 DPAS

见 [qsa_prepare.cm](kernels/qsa_prepare.cm)。每个 WG 拥有一个 compression block，WG16；循环处理该 block 中的新 token。激活先进入 SLM，但对每个投影输出维度，仍重新读取权重行和 SLM 激活，做 FP32 乘法与 `cm_sum`。

投影宽度 `J=(Hidx+1)*Di=640`，每 token 约 3.277 MFLOPs，FP16 权重约 3.125 MiB。当前虽然名为“CM tiled GEMM”，**源码没有 `cm_dpas`**。相邻 token 没有通过矩阵 tile 显式复用同一权重 tile；缓存可能命中，但不能消除指令和读取请求。

推荐：

- Prefill：将多 token 的 `[T,2560] × [2560,640]` 做成真正的 DPAS GEMM，或先用成熟 GEMM 建立性能基准；norm/RoPE/pooling/store 做小型后处理。
- 不必追求一个巨大融合 kernel。FP32 投影中间结果每 token 仅 2560 bytes，写入再读回约 5120 bytes，可能比现有重复投影访存/规约便宜得多。
- Decode：单 token 不适合照搬大 prefill tile；优先比较多 WG GEMV、按输出 tile/K 维拆分，以及微批量 DPAS 路径，避免只有一个 WG 做全部 640 个输出。
- Dense-prefix key-only：不需要 selection 时，省掉 512 维 indexer Q 投影，仅保留 128 维 indexer K 投影和状态更新。理论上省掉该部分 80% 的投影 FLOPs，不等于 Q1 总耗时降低 80%。

数值风险：当前新 raw K 以 FP32 留在 SLM 用于 pooling，同时以 FP16 写 cache；跨轮旧 raw K 则从 FP16 cache 读取。重构若把所有中间投影先转成 FP16，会改变这一舍入边界。应优先保留 FP32 中间值，并验证跨 block/chunk 的 summary、scores 及 top-k 边界，而不是只检查最终 attention 的宽松误差。

### 4.2 Q2 top-k：每个 query 只有一个硬件线程

见 [qsa_common.hpp](include/qsa_common.hpp) 中 `radix_select_topk` 和 [qsa_topk_finalization.cm](kernels/qsa_topk_finalization.cm)。

当前使用 4 轮 8-bit radix histogram，每轮扫描完整有效分数范围，最后升序扫描一次完成输出和 tie 处理。256-bin histogram 由同一硬件线程持有/处理；没有 `n_valid <= k` 的直接顺序输出快速路径。

这不是 `O(n log k)` 的堆排序，也不是已经无可改进的“最优 top-k”。

推荐分两步：

1. **先做 exact dense bypass**：top-k 内部直接输出 arange；随后在 score 调度处跳过这些 rows，才能连 score 一起省掉。
2. **再做 cooperative radix**：一个 WG 协作一行，分块读分数、局部 histogram、归并 histogram，最后用 prefix-sum 按 block ID 有序压缩输出。对长行可缩小候选集，但需可靠的溢出 fallback。

保持同分时按 block ID 的既有规则。不能将 unordered atomic append 结果直接送入 CM Q3 的归并；对于 ReLU 后大量零分、全相同分、阈值附近极小差值，要单独测试。

### 4.3 Q2 decode 的 partition 名称掩盖了并行度不足

见 [qsa_score_partition.cm](kernels/qsa_score_partition.cm) 和 driver。每个 partition 扫 512 summaries，`lws=1`。batch1、past4096 只有 1024 summaries，因此只有 **2 个 work-items** 做 scoring。

推荐将粒度改成 16/32/64 summaries 的 tile 并测试，或用 WG 协作/DPAS 做 block scoring。必须先完成每个 indexer head 的 dot，再分别 ReLU，最后求和；不能把不同 head 的 partial dot 提前相加。

SGLang decode score 按压缩页组合成 64-summary GEMM tile、沿 block 维并行，是可迁移的组织思路；其 CUDA 线程数和 compressed-page 限制不是 CM 的参数标准。

### 4.4 Q2 长上下文 workspace

当前 DPAS score 与 top-k 分开，保留 `[num_tokens,max_blocks]` 的 FP32 scratch；scalar `score_topk_fused` 也先写该 scratch 再做 radix，**融合 launch 不等于消除了分数矩阵**。

单序列完整 prefill、r=4 时，矩形 scratch 约为 `N*N` bytes：

| N | FP32 score scratch |
|---:|---:|
| 4096 | 16 MiB |
| 32768 | 1 GiB |
| 65536 | 4 GiB |
| 131072 | 16 GiB |

实际插件 buffer 上界还按整个物理 cache 池计算，而不只是当前最长序列，可能更大。

推荐借鉴 SGLang：按 query rows 分块，让 score→top-k 消费完成后复用 scratch，预算可从 128 MiB 起测。该方案不改变完整历史候选集，也不意味着把 top-k 改成每个 history chunk 独立选完就结束。若沿候选列分块，则需要正确合并局部候选到全局 top-k。

Q2 score 本身的次级机会包括：summary 2D load 从 16 half 列扩到合法的 32 half 列；跨 block tile 改善 Q 的 SLM 复用。不要因为“支持 DPAS”就推定访存已最优，但本轮 1K–4K 数据表明这些不应排在 Q1/top-k 前面。

## 5. Q3 的下一步：减少重复工作，而不仅加宽 load

### 5.1 当前映射的得失

见 [qsa_sparse_attention_dpas.cm](kernels/qsa_sparse_attention_dpas.cm)。默认 `HT=4, TT=4, W=4`：一个 DPAS N 轴是 4 heads × 4 tokens，W 个线程划分 head dimension。

优势：复用同一 KV 给 4 heads；相邻 query 选择重合时可复用它们的 KV；直接读取 paged cache，无 selected-KV 实体化。

剩余问题：

- 对同一个 kv_head 和同一个 4-token subtile，12 个 query heads 分成 3 个 head groups，各自加载相同并集 KV，产生 **3 份逻辑 KV 读取**。W4 切的是维度，不应再错误乘成 12 份完整 KV。
- 并集 metadata 的归并则在 6 个 head groups × W4 中重复，最多是每个 subtile 的 24 份相同控制计算。
- 每次 QK partial score 需要 SLM 交换；增加 W 会增加规约与同步代价。

物理页表打乱不妨碍同一逻辑页内部的连续 block 合并。上轮优化正是利用了这一点。

### 5.2 长上下文的 union amplification 不能忽略

设 4 个 query 各从 B 个完整 block 中独立等概率选 K 个，U 为它们的并集大小，则：

$$\mathbb{E}[U]=B\left[1-(1-K/B)^4\right].$$

这是 synthetic 独立选择的模型，不是实测模型 token 的重合度；也不是一次完整 prefill 各行的平均值。

| B | K | E[U]/K |
|---:|---:|---:|
| 512 | 512 | 1.000 |
| 1024 | 512 | 1.875 |
| 2048 | 512 | 2.734 |
| 8192 | 512 | 3.640 |

当历史远大于预算时，该值趋近 4。DPAS 仍要对 union 上的列计算，即便随后 mask 掉某个 query 未选择的位置。不能把 `top-k=512` 一概当作低 union 放大的 dense 场景。

### 5.3 两条可比较的精确 Q3 路线

**路线 A：保留 4-token tile，将 3 个 head teams 放入同一个 WG，共享 KV staging。**

- 例如 3 teams × W4 = 12 个 CM threads，共享同一 kv_head 的 K/V tile；每个线程保留原有输出 accumulator 分片。
- 不能跨 WG 共享 SLM；也不应让一个线程直接承担原来三倍的 accumulator。
- 16-position 完整 K+V tile 为 16 KiB，双缓冲 32 KiB；三组各自的双缓冲 partial St 合计约 24 KiB，总计约 56 KiB，尚未计其他 metadata、对齐及实现开销。
- 这是容量可行性估算，不是 occupancy 保证。需检查实际 SLM 分配、barrier 代价和寄存器；减少全局读取可能增加 SLM 流量并降低驻留 WG 数。

**路线 B：SGLang 式一个 query × 同一 KV 的全部 12 heads，DPAS head 轴补到 16。**

- 消除跨 token union，但 16 列中只有 12 列有效，利用率为 75%。需要新的布局/dispatch，不能只把当前 HT 改成 12 或 16。
- 忽略尾部、padding 及缓存后，对相同 4 个 query / 一个 KV head：旧路径逻辑 KV 为 `3U`，新路径为 `4K`。当 `U/K > 4/3` 时，新路径才在这个 KV 读取模型上更少；完全重合的 dense 选择反而可能不划算。
- 更合理的目标是 dense/high-overlap 用多 token tile，低 overlap 用全 GQA heads tile，而不是一种 kernel 覆盖全部情况。

两条路线都应与现有 optimized kernel 比较，不用上轮 baseline 做偏宽松的对照。3 倍冗余读取并不意味着 3 倍性能上限，更不能直接推导实际会快 1.5–2 倍。

### 5.4 先共享 metadata，未必先共享全部 KV

可先增加一个轻量 prepass，为每个 sequence-local subtile 生成 union IDs、token membership mask，必要时加 page ID。其结果可由所有 head groups/worker 复用，消除重复归并和页表推导。

代价是额外 launch、metadata scratch 和读写。每条 union metadata 若占 8–16 bytes，总空间随 subtile 数 × U 增长；应 chunk 化复用，避免又引入大 workspace。短序列保留 inline merge 可能更好。

### 5.5 Decode：split selected-KV，不是继续增加 head-dimension workers

当前 [qsa_sparse_attention.cm](kernels/qsa_sparse_attention.cm) 每个 `(query,head)` 一个 CM work-item，batch1 仅 24 个，顺序处理最多 512 blocks；与 160 EUs 不匹配。

推荐按 selected block 列表分 P 份，每份独立 online softmax，再做合并。可从 `P=1,2,4,8,16` 对比；也可与全 GQA-heads DPAS 映射组合。

若分区 p 输出局部最大值 m_p、指数和 l_p、未归一化加权和 a_p，则：

$$m=\max_p m_p,\quad l=\sum_p e^{m_p-m}l_p,\quad o=\frac{\sum_p e^{m_p-m}a_p}{l}.$$

tail 只能归属一个 partition；空 partition、全空输入必须避免 `-inf - (-inf)` 导致 NaN；output gate 应放在最终归一化之后。

小预算/大 batch 可能已经有足够并行度，额外 reduction 反而更慢。分区数必须随 batch、selected_count 和资源配置选择，而非固定越大越好。

## 6. SGLang：可迁移策略与不可照搬之处

| 实际源码 | 已确认策略 | 对 CM 的意义 |
|---|---|---|
| [qsa_indexer.py](../sglang/python/sglang/srt/layers/attention/qsa/qsa_indexer.py) | ReplicatedLinear 投影；独立 fused prep/compress；持久化 summary；prefill score workspace 128 MiB 分块 | GEMM 与轻量后处理分离；bounded scratch；不能因拆 kernel 就预判更慢 |
| [mqa.py](../sglang/python/sglang/srt/layers/attention/qsa/mqa.py) | Prefill query/head 打包 GEMM，64-key tiles，pipeline；decode 压缩子页组合为 64-row tile | 批量 scoring、较细 block 并行；本配置 compressed page4 不符合其 decode wrapper 的 8/16/32/64 限制，需 CM 自行适配 |
| [fast_topk.cuh](../sglang/python/sglang/kernels/jit/csrc/elementwise/fast_topk.cuh) | length<=K 直接输出；1024 CUDA threads/row；粗分桶、候选压缩、radix refinement | 学 cooperative selection，不照搬线程数、tie/output 顺序及 scratch 假设 |
| [sparse_attn.py](../sglang/python/sglang/srt/layers/attention/qsa/sparse_attn.py) | Prefill 一个 query/kv_head，GQA heads 补到至少16；KV chunk 复用于这些 heads；在线 softmax | 对应 Q3 路线 B，避免跨 query union |
| [qwen_sparse_attn_backend.py](../sglang/python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py) | Decode 可 pack selected KV 后调用 TRTLLM/FlashInfer 或 varlen FlashAttention；MTP draft 可复用 captured selection | 成熟 decode backend 的复用路线；MTP 仅是特定运行策略 |

重要限制：

- fast-topk 输出顺序不保证升序；CM 必须有序压缩或补排序。其当前候选缓冲有 4096-entry 容量及截断处理，不能把固定容量假设直接移植为任意分数分布下的精确算法；CM 应对大阈值桶/溢出重扫并做确定性 tie 处理。
- SGLang 普通 prefill 路径与带 prefix 路径不同：带 prefix 的现有 CUDA 路径会 gather 整个 context K/V 后运行 chunk-prefill。CM 的直接 paged gather 在这一点上已有优势，不必照抄全 context 重排。
- 对本配置，2051 个 selected token 的 FP16 K+V 约为 `2051*2*(256+256)*2 ≈ 4 MiB/query`。pack 的读源、写临时、attention 再读临时，理想逻辑搬运就约 12 MiB/query，尚未计 backend 内部重读。它可能优于当前 scalar 的重复读取，但绝不是“零成本合并访存”，尤其不宜盲目用于大 prefill。
- MTP draft reuse 有独立的捕获/更新语义。不能把普通连续 decode 的 query 都复用同一选择集，当作精确的 kernel 优化。

## 7. llama.cpp：哪些是真实 QSA 优化

### 7.1 当前主 attention 仍是 dense-mask 路径

[qwen4exp.cpp](../llama.cpp/src/models/qwen4exp.cpp) 的 `build_qsa_top_k` 已有 indexer 投影、pool/norm/RoPE、score、top-k；但 `build_attn_qsa` 明确包含 `TODO: enable sparse attention when we are ready`，实际 `build_attn_mha` 的 sparse hint 传 0。

当前通过构造稠密 `-inf` mask、在选中位置写入允许值来限制 attention；不能把它称为已启用的 direct selected-KV Q3。通用 FA 后端可能跳过全 mask tile，但不等价于保证 O(K) 的稀疏访问复杂度。

其 indexer 图还会从 raw K cache gather block 成员并构造 pooled K；不能把它当成 CM/SGLang 持久化 summary 的更优替代。

### 7.2 Vulkan 的 QSA top-k 融合确实存在，但有 scratch

见 [topk_radix_select.comp](../llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/topk_radix_select.comp) 和 `ggml-vulkan.cpp` 中 `ggml_vk_topk_qsa`。

一个 WG 协作一行，shared histogram + atomicAdd 做 radix。QSA specialization 在首次扫描中按 cell→block 映射 gather score 并加 mask，**同时写入 scratch，后续扫描读取 scratch**。因此它融合了图上的 gather/mask 处理，不是完全没有展开分数或临时内存。

输出使用 atomic append，顺序及 ties 不能直接满足 CM 升序契约；值得迁移的是 cooperative histogram 和减少独立图节点，而不是逐行照抄 shader。

### 7.3 通用 FlashAttention split-K 是有价值的参考

[flash_attn_split_k_reduce.comp](../llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_split_k_reduce.comp) 展示了用各分区 max/sum/加权输出合并 attention 的实现。可参考其组织方式设计 CM decode split-selected-KV，但这是通用 FA 技术，不能据此声称 llama.cpp 已在此模型中启用真正的 sparse Q3。

## 8. 运行时与 GR 的边界

[OpenVINO qsa.cpp](../openvino.mx/src/plugins/intel_gpu/src/graph/impls/cm/qsa/qsa.cpp) 已确认：

- Q0/Q1 使用相同输入依赖，两者不串成相互依赖；Q3 等待 KV 与 selection。
- Prefill/mixed 在支持配置上使用 DPAS score + finalization、DPAS Q3；GENERATE 使用 scalar partition score + finalization、scalar Q3。
- `update_rt_params` 读取 past/subsequence metadata，构造 prepare/tile maps；执行时上传 maps。短 decode 应额外测量这些 CPU/同步成本，并考虑调度器共享元数据或 GPU 构图，避免逐层重复工作。
- 若把 dense/sparse rows 分开调度，不能只按整个 batch 的 mode 判断；混合 batch 需要 sequence-local 分类和正确依赖。

当前 standalone 目录不包含 GR read/write 或主 QKV/o_proj GEMM 的完整链路，因此这里没有 GR 端到端性能结论。后续模型级 profiling 应覆盖 GR、main projections、QSA、o_proj；可考虑投影 epilogue 与 cache write/gate 等融合，但不能通过提前应用 gate 或改变 GR mix 顺序来节约计算。

## 9. 推荐实施顺序

| 优先级 | 工作 | 原因 | 验证重点 |
|---|---|---|---|
| P0 | 接通真实 Q1→Q2→Q3 benchmark；记录阶段和 CPU 耗时 | synthetic selection 不能代表模型重合度 | warm/cold、依赖、真实选择、P50/P95 |
| P1 | Dense rows 的 exact score/top-k bypass；可选 Q1 key-only | 小改动即可删除无用工作，1K/2K 全覆盖 | 可见长度2051/2052边界、尾部、未来 summary 状态 |
| P1 | Q1 prefill 真 DPAS GEMM；decode 多 WG GEMV | 本轮最大 prefill 瓶颈；decode 也只有一个 WG | FP32 中间值、norm/RoPE、跨 chunk pooling |
| P1 | Q2 cooperative top-k + decode 细粒度 scoring | top-k 比 DPAS score 慢一个数量级以上 | 升序、ties、溢出 fallback、选中集合 |
| P1（decode） | Q3 split-selected-KV / grouped-head DPAS | 24 work-items 与 GPU 并行度不匹配 | 归一化合并、空分区、P 的 break-even |
| P1（长上下文） | Q2 bounded row workspace | 防止 score scratch 达 GiB 级 | 多序列、chunk row offset、并发 workspace 复用 |
| P2 | Q3 metadata prepass / 同 WG 跨 head-team KV 复用 | 减少重复归并与 3 份 KV 请求 | SLM/GRF/occupancy/barrier 及额外 launch |
| P2 | Q3 overlap-aware token/head tiling + dense causal fast path | 避免长上下文 union 放大 | dense、高重合、随机、近不重合选择 |
| P2 | Q2 32-half summary loads、Q0/projection epilogue 融合 | 局部优化，排在大瓶颈之后 | page边界、layout、寄存器与依赖 |
| P3 | selected-KV packing、独立 summary 布局 | 较大设计变化，是否划算取决于复用 | pack+attention 总成本、cache生命周期 |
| 单独项目 | KV量化、近似top-k、跨query/MTP复用策略 | 涉及数值或模型行为，不是纯精确kernel改写 | 模型质量、acceptance/困惑度及完整回退 |

如果只选下一项较大 kernel 工作：**prefill 优先 Q1 DPAS 投影；decode 优先协作式 Q2 与 split-selected-KV Q3。** 并行推进低风险的 dense bypass，而不是继续只对 Q3 增加 prefetch 或盲目扩大 W/HT。

## 10. 验证与性能模型修正

### 正确性

- 独立验证 indexer Q、raw K、summary、score、top-k 集合/顺序、Q3 输出；score 小误差不自动保证 top-k 相同。
- 覆盖 token 数 0/1/3/4、可见长度2047–2053、长 history、batch1/4/16、mixed prefill/decode、空序列、跨页及跨压缩块的 chunk。
- top-k 覆盖全相等、零分、阈值 tie、大候选桶、K=0/1/512、C<K/C=K/C>K；不能只测均匀随机分数。
- 每个新 Q3 路径沿用现有 18-case 回归，并扩展 split-K/新 tile 支持范围；改变计算顺序的方案不强求逐 bit 相同，但误差、边界选择和模型影响必须解释。
- 多数现有 standalone driver 的 `--check` 只打印 FAIL 而不抛异常，自动化验收前应统一成失败返回非零；本轮阶段计时没有把这些 driver 当作严格全链正确性验收。

### 性能

- 最终验收采用 warmup100、同输入/同布局/同缓存条件、交替 A/B、多轮采样，记录 mean/median/P95；同时报告不 flush 的集成时延。
- 记录真实 selected-set 的 U/K、page 连续率、有效 DPAS 列比例；不能只用独立随机 top-k 判断 tile 优劣。
- 检查编译后 GRF、scratch/spill、SLM、load 指令数、barrier；有条件再采集硬件计数器，区分请求数、缓存流量和 DRAM bytes。
- `qsa_work.q1` 的 FLOPs 可作有用工作量，但当前 scalar Q1 不应直接对照 XMX 峰值解释“计算利用率”。
- `qsa_work.q2_topk` 的简化流量没有完整反映 radix 多次读取；融合路径也有 scratch 流量。
- `qsa_work.q3` 用最大位置估计 union，不等于实际 sparse union；多序列也不能只取全局最大位置。`profile_q3_dpas.py` 的分组需按 sequence 边界并计入 tail。
- 逻辑等效带宽高于 456 GB/s 不是缓存命中率证据，更不能单凭它认定 message-issue-bound。旧分析中的固定 1.5–2x 上限、长上下文 union 始终很小等结论不再适用。

**最终判断：当前实现距离“只剩微优化”还很远。已有 Q3 load 优化应保留，但下一轮收益更应从执行真正的矩阵投影、删除无需选择的工作、增加 decode 并行度，以及减少跨 heads/tokens 的重复计算中获取。**