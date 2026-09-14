# QSA Kernel 算法与优化总结

## 1. 文档范围

本文总结 standalone `cm_kernel` 中 QSA（Query-centric Sparse Attention）从当前 token、indexer 状态更新，到候选 block 选择，再到稀疏主 attention 的完整算法。重点说明：

- Q0：主 attention K/V 写入 paged KV cache；
- Q1：Indexer Q/K 投影、归一化、RoPE、raw K 与 compressed summary 生成；
- Q2：Indexer summary 打分、精确 top-k block 选择；
- Q3：依据 selected blocks 读取主 K/V，执行稀疏 attention；
- prefill 与 decode 的不同 kernel 组织；
- 当前保留的性能优化、精确性约束和未覆盖的外围计算。

本文讨论的是 standalone QSA kernel slice，不等于完整模型端到端执行。当前 slice 不包含外部 GR、主 QKV/o_proj、调度器、模型层间同步和完整运行时开销。

---

## 2. 模型几何与核心符号

当前 `config.json` 解析得到的默认几何如下：

| 符号 | 含义 | 当前值 |
|---|---|---:|
| `D` | hidden size | 2560 |
| `Hq` | 主 attention query heads | 24 |
| `Hkv` | 主 attention KV heads | 2 |
| `Hq/Hkv` | GQA replication ratio | 12 |
| `Dh/Dv` | 主 attention K/V head dimension | 256 / 256 |
| `Hidx` | indexer heads | 4 |
| `Di` | indexer head dimension | 128 |
| `rot` | indexer partial RoPE dimension | 32 |
| `r` | token 到 compressed block 的压缩率 | 4 |
| `budget` | token budget | 2048 |
| `Kb` | block top-k，约为 `ceil(budget/r)` | 512 |
| `PA` | paged-attention page token 数 | 16 |
| `PA/r` | 每个 page 中的 summary 数 | 4 |

对绝对位置 `t`，可见的完整 compressed block 数为：

$$
C_t = \left\lfloor\frac{t+1}{r}\right\rfloor.
$$

QSA 选择最多 `Kb=512` 个完整 block，然后额外保留当前 query 末端尚未形成完整 block 的尾部 token。因此最多访问：

$$
Kb\times r + (r-1) = 512\times 4 + 3 = 2051
$$

个主 KV token。

一个重要边界是：当 `C_t <= Kb` 时，所有完整 block 都可直接选中，不需要真正做 score/top-k；对于当前配置，dense causal 区域延伸到可见长度 2051 附近，而不是简单地把 2048 当作严格边界。

---

## 3. 总体数据流

QSA 的执行顺序可以抽象为：

```text
外部当前 token 的主 K/V、GR 输出、位置和 page map
                    │
                    ├── Q0：写入 paged main K/V cache
                    │
                    └── Q1：Indexer Q/K projection
                              │
                              ├── indexer Q：norm + token-position RoPE
                              ├── raw indexer K：写 raw-K cache
                              └── 完整 block：FP32 pooling + norm + block RoPE
                                                │
                                                └── summary cache
                                                         │
                                  Q2：Q · summary score → exact top-k
                                                         │
                              selected block IDs + count + causal tail
                                                         │
                                  Q3：selected main K/V attention
                                                         │
                                             output gate 后的 QSA 输出
```

Indexer 分支和主 attention 分支使用不同数据：

- indexer Q/K 用于决定哪些 compressed blocks 值得访问；
- 主 attention Q/K/V 使用原始主 attention 数据；
- summary 只用于 Q2 选择，不替代 Q3 中的主 K/V；
- 同一个 query 的 selected block 集通常由所有主 attention heads 共享，但不同 query token 通常有不同选择集。

---

## 4. Q0：主 K/V 写入 paged cache

### 4.1 算法

Q0 kernel 为 `qsa_kv_cache_update.cm`。它把外部已经投影好的当前 token K/V 写入 paged main KV cache：

- 一个 work-item 对应一个 `(token, kv_head)`；
- 根据 `past_lens` 和 `subsequence_begins` 找到 token 所属 sequence；
- 计算绝对位置 `abs_pos = past_len + local_token_offset`；
- 使用 `block_indices` 将逻辑 page 映射到物理 page；
- 根据 `token_in_page` 计算 page 内偏移；
- 分别将完整的 `Dh=256` K head 和 `Dv=256` V head 写入 cache。

主 KV cache 的逻辑布局可表示为：

$$
K/V[\text{physical page},\ \text{KV head},\ \text{token in page},\ \text{head dimension}].
$$

Q0 不执行数学变换，也不做压缩、归一化或 RoPE。它是纯 copy/update kernel，主要优化点是使用宽向量 load/store 减少 LSC message 数。

### 4.2 为什么 Q0 不与 Q1 融合

Q0 的自然并行单位是 `(token, KV head)`，而 Q1 的自然并行单位是 `(sequence, compression block)`。强行融合会带来：

1. Q1 依赖的是 indexer projection 和 summary 状态；
2. Q3 依赖 main KV cache 完成写入；
3. mixed batch 中不同 sequence 的 token/block 边界不同；
4. 一个 Q1 work-group 可能处理多个 token，而 Q0 是单 token 写入。

因此当前设计保持 Q0 独立，使 main KV 写入依赖清晰，也避免为 prefill/decode/mixed batch 引入更多融合变体。

### 4.3 性能特征

Q0 算术强度接近零，主要受内存消息和地址计算影响。当前 PTL 长上下文测试中，Q0 不是主要瓶颈：prefill 长上下文的总时间主要由 Q3 和 Q2 top-k 主导；decode 中 Q0 也远小于 Q1/Q2/Q3。

---

## 5. Q1：Indexer projection、压缩与 summary 构造

Q1 由两部分组成：

1. 可选的 `qsa_project_dpas.cm`：只做 `[tokens,D] × [D,J]` projection；
2. `qsa_prepare.cm`：对 projection 结果做 Q/K 后处理、cache layout 和 summary 生成。

其中：

$$
J=(Hidx+1)\times Di=(4+1)\times128=640.
$$

640 个输出通道包含 4 个 indexer Q heads 和 1 个 raw indexer K。

### 5.1 Q1 projection

对每个 token，先计算：

$$
P_t = W_{idx}\,x_t,
$$

其中 `x_t` 是 hidden/GR 输出，`W_idx` 的输出宽度为 `J=640`。当前 DPAS projection 使用：

- 一个 work-item 负责一个小的输出/Token tile；
- `N=16` 个 token 作为 DPAS 的 query/token 维；
- `M=8` 个输出通道作为 DPAS 输出维；
- `K=16` half 元素作为每次 systolic 累加步长；
- 累加结果保存为 FP32；
- 最后转换为 token-major `[token,J]` 的 FP32 projection buffer。

投影使用 FP16 输入/权重，FP32 累加；这是后续 pooling 精度的重要基础。

`qsa_prepare.cm` 也保留了不使用独立 DPAS projection 的路径。该路径将每个 token 的 activation 放入 SLM，再用 tiled CM 乘加计算投影。当前推荐的 prefill 路径优先使用独立 DPAS projection，再让 prepare 使用 `QSA_PREPROJECTED` 读取 FP32 projection。

### 5.2 Indexer Q：token 级 norm + RoPE

对 projection 中的每个 indexer Q head：

1. 取出 `Di=128` 个投影值；
2. 执行 RMSNorm；
3. 在 query 自身绝对位置 `t` 应用 partial RoPE；
4. 转为 FP16 写入 `indexer_q[token, head, dim]`。

数学上可写为：

$$
q^{idx}_{t,h}=\operatorname{RoPE}_{t}
\left(\operatorname{RMSNorm}(P_{t,h}^{Q})\right).
$$

顺序不能交换：必须先 RMSNorm，再 partial RoPE；RoPE 位置是 token 的绝对位置，而不是 compressed block 编号。

### 5.3 Raw indexer K：只做 cache layout transform

raw indexer K 不在 token 写入阶段做 RMSNorm 或 RoPE：

$$
rawK_t = P_t^K.
$$

它以 FP16 写入 raw-K cache，但刚生成的 FP32 raw K 同时保存在 SLM 中，以便后续 block pooling 直接使用新鲜的 FP32 值。

这一区分非常重要：

- 新 token 的 raw K：pooling 使用 FP32 projection；
- 跨 dispatch 读取的旧 raw K：从 FP16 raw-K cache 读取；
- 不能把新 raw K 先写成 FP16，再读回 FP16 做 pooling，否则会改变舍入边界。

### 5.4 完整 compressed block 的 summary

只有当一个 compression block 的 `r=4` 个 token 全部可用时，Q1 才生成 summary。对 block 起点 `b`：

$$
\bar{k}_b = \frac{1}{r}\sum_{i=0}^{r-1} rawK_{b+i}.
$$

pooling 使用 FP32 累加。若 block 内同时包含历史 token 和新 token：

- 历史槽位从 raw-K cache 读取；
- 新槽位优先从当前 work-group 的 SLM 读取；
- 不依赖刚刚写入的 global memory 再读回。

随后执行 block-key 的 RMSNorm 和 RoPE：

$$
summary_b = \operatorname{RoPE}_{b}
\left(\operatorname{RMSNorm}(\bar{k}_b)\right).
$$

这里的 RoPE 位置是 **block 起点 `b`**，不是 block index，也不是 block 末端位置。

summary 按 paged layout 写入 `summary_cache[physical_page, summary_slot, Di]`。由于一个 PA page 包含 `PA/r=4` 个 summary，summary 的物理地址计算与 page map 一致。

### 5.5 不完整尾部 block

未满 `r` 个 token 的尾部 block 不产生 summary。Q3 会把尾部 token 作为 always-visible tail 直接追加访问。

例如当前位置使得最后一个完整 block 之后还剩 1、2 或 3 个 token，Q2 只选择完整 block，Q3 额外访问这些尾部 token。这样可避免用不完整 block 的平均值代表真实 token。

### 5.6 Q1 的主要优化方向

当前 Q1 的主要问题是 prefill projection 工作量大。推荐方向：

- prefill 使用真正的 DPAS/GEMM projection；
- 将 norm/RoPE/pooling/store 保留为轻量后处理；
- decode 避免把大 prefill tile 生搬硬套到单 token，比较多 work-group GEMV、输出 tile 拆分和微批量 DPAS；
- dense causal query 可以跳过 indexer Q 的无用 selection 投影，但仍必须生成 indexer K 和 summary 状态。

注意：dense bypass 不能跳过 raw K/summary 更新，因为未来 query 仍需要这些历史 summary。

---

## 6. Q2：Indexer score 与精确 block top-k

Q2 的输入是：

- 当前 query 的 `indexer_q`；
- 历史 compressed `summary_cache`；
- page map 和 sequence metadata。

输出是每个 query 的：

- selected block IDs；
- selected count；
- 升序排列的 selected IDs；
- 未使用位置填充 `-1`。

### 6.1 Score 数学

对 query `t`、compressed block `b`、indexer head `h`：

$$
 s_{t,b,h}=q^{idx}_{t,h}\cdot summary_{b,h}.
$$

QSA 的 score 不是普通的跨 head dot product。必须保持：

1. 每个 indexer head 独立计算 dot；
2. 对每个 head 单独执行 ReLU；
3. 再跨 indexer heads 求和；
4. 最后乘 `1/sqrt(Di)`。

$$
I_{t,b}=\frac{1}{\sqrt{D_i}}
\sum_{h=0}^{Hidx-1}\operatorname{ReLU}(s_{t,b,h}).
$$

ReLU 的位置不能移到 head reduction 之后，因为：

$$
\sum_h \operatorname{ReLU}(s_h) \neq
\operatorname{ReLU}\left(\sum_hs_h\right).
$$

### 6.2 Prefill score：DPAS tile

prefill 使用 `qsa_score_tile_dpas.cm`：

- 一个 work-group 处理最多 16 个 query；
- 每个 score tile 处理多个 compressed blocks；
- query tile 先放入 SLM，供所有 block tile 重复使用；
- summary tile 通过 page-aware 2D load 读取；
- DPAS 沿 `Di=128` 的 K 维累加；
- 每个 indexer head 的 DPAS 结果先 ReLU，再累加到总 score；
- 最终转置为 query-major score row 写入 scratch。

该 kernel 只负责 score，不负责最终 top-k。这样 prefill 与 decode 共用 selection 语义，避免 work-group 内刚写 global score 后又被其他 lane 读取造成 coherence/fence 风险。

### 6.3 Decode score：partition

decode 由于只有一个新 query，使用 `qsa_score_partition.cm`：

- 历史 summary 被划分为多个 partition；
- 每个 partition 扫描一段 summary；
- 计算结果写入 query 的 score row；
- 后续统一进入 top-k finalizer。

partition 的优点是可以让 decode score 的历史扫描具有更多并行度；但 batch=1 时，partition 数和 work-item 粒度仍然很关键。当前实现使用 `partition16` 优化路径，短上下文或不满足条件时回退到其他路径。

### 6.4 Dense bypass

若一个 query 的完整 block 数不超过 `Kb`，所有完整 block 都会被选中：

$$
selected=[0,1,2,\ldots,C_t-1].
$$

此时可以：

- 直接生成顺序 block IDs；
- 不读取 score row；
- 不运行 top-k；
- 保持 scratch 为 poison/原值，避免错误读取未写入的 score。

在混合 tile 中，dense 和 sparse query 可能同时存在。kernel 可以统一 dispatch，但只对 sparse row 发布 score；dense row 必须保持其 scratch 不被误用。

### 6.5 Exact top-k finalizer

当前 top-k 支持 legacy、cooperative、fast、fast-WG 和 `fast-wg16-cached` 等模式。共同语义是精确选择，不能接受近似漏选。

排序 key 使用 ordered floating-point bits，使有限 FP32 score 可进行 radix ordering，同时保留：

- signed-zero 的确定性顺序；
- 相同 score 时按 block ID 升序 tie-break；
- 输出 selected IDs 升序；
- K=0、K=1、候选数小于 K、大候选数的正确性。

`fast-wg16-cached` 的优化是：

1. 每个 worker 一次加载自己的 SIMD64 ordered keys；
2. threshold selection、greater/equal 统计和最终输出重复使用缓存；
3. 当候选宽度超过缓存容量时整行回退到 exact global scan；
4. 不截断候选，也不改变 tie-break。

容量为：

$$
capacity=\min\left(2048,\left\lceil\frac{max\_blocks}{1024}\right\rceil\times64\right).
$$

例如：

- 64K history：最大 score width 16384，容量 1024 keys/worker；
- 128K history：最大 score width 32768，容量 2048 keys/worker；
- 超过容量的 row：精确回退，不是近似截断。

### 6.6 Q2 scratch 与长上下文

如果为完整 prefill 一次性分配 `[num_query,max_blocks]` FP32 score matrix，`r=4` 时内存近似为：

| Query length | score width | FP32 scratch |
|---:|---:|---:|
| 4K | 1K | 16 MiB |
| 32K | 8K | 1 GiB |
| 64K | 16K | 4 GiB |
| 128K | 32K | 16 GiB |

当前使用 query-row chunking：

- 一个 chunk 只包含有限 query rows；
- score 完成后立刻执行 top-k；
- 下一个 chunk 重用同一 scratch；
- score row 不跨 sequence/chunk 边界；
- guards、poison 和 row offsets 用于发现越界及错误复用。

这减少了工作空间，但没有把候选列切成独立 top-k，也没有丢弃全历史候选。

---

## 7. Q3：selected main K/V 的稀疏 attention

Q3 使用 Q2 的 selected block IDs 访问主 attention K/V cache。summary 只用于选择，Q3 仍然读取真实 main K/V。

### 7.1 Attention 数学

对主 query head `a`，selected token 集合为 `S_t`。对每个 selected token `j`：

$$
 z_{t,a,j}=\frac{Q_{t,a}\cdot K_{j,a}}{\sqrt{D_h}}.
$$

causal sparse attention 只保留：

- 选中的完整 block 中的 token；
- 当前尾部不完整 block 的 always-visible token。

然后执行 online softmax：

$$
 m=\max_j z_j,
\qquad
 l=\sum_j e^{z_j-m},
\qquad
 o=\frac{\sum_j e^{z_j-m}V_j}{l}.
$$

对于输出 gate，主 attention 结果在 QSA 内按配置执行 sigmoid gate；GR read/write gate 和 `o_proj` 属于外部模型链路，不属于 standalone Q3 kernel。

### 7.2 GQA DPAS prefill

当前保留的 prefill kernel 是 `qsa_sparse_attention_gqa_dpas.cm`：

- 一个 work-group 对应一个 query token 和一个 KV head；
- 一个 KV head 服务其 GQA replication 的 12 个 query heads；
- `W4` 将 head dimension 划分成 64-dimension slice；
- 每次处理 16 个 selected token；
- partial score 通过双缓冲 SLM 交换；
- FP16 DPAS 与 online softmax 算术顺序保持不变。

这一路径没有跨 token union。不能将当前 kernel 错误地解释为旧 HT4/TT4 union kernel。

### 7.3 当前保留的 Q3 优化

#### KV-head-major 调度

默认 work-group 映射会交错 KV heads。优化后交换 NDRange 轴，使相邻工作组更集中于同一个 KV head：

- token 轴变为更快变化的轴；
- KV-head 轴变为较慢变化的轴；
- 主 K/V 地址、结果地址和算术不变；
- 目标是改善相邻 work-group 的 cache/TLB/page locality。

这是针对长上下文离散 KV gather 的调度优化，不改变算法结果。

#### Vector metadata

原始路径逐个读取 selected ID 和 physical page ID。优化后：

- 将每 step 的 selected IDs 组织成小向量；
- 对有效 lane 执行 masked gather；
- page IDs 也使用 masked gather；
- 尾部只使用一个有效位置；
- 无效 lane 不访问 metadata。

它减少地址链和 metadata load 的开销，但单独启用收益小，和 head-major 组合时效果最好。

### 7.4 Decode split-selected-KV

decode batch=1 时，传统 scalar Q3 的并行度只有 query heads 数量，难以填满 GPU。当前优化路径将 selected blocks 按 partition 拆分：

1. 每个 partition 独立扫描一部分 selected KV；
2. 每个 partition 计算局部 online-softmax 状态：`m_p`、`l_p`、`a_p`；
3. reducer 合并所有 partition：

$$
 m=\max_p m_p,
$$

$$
 l=\sum_p e^{m_p-m}l_p,
$$

$$
 a=\sum_p e^{m_p-m}a_p,
\qquad
 o=a/l.
$$

空 partition 必须安全处理，不能产生 `-inf - (-inf)` 的 NaN；尾部 token 只能归属一个 partition。小 batch/短 selected list 不一定适合 split，当前 dispatch 只在满足形状条件时启用。

### 7.5 Q3 长上下文瓶颈

每个 query 的 selected 数量上限固定，但 KV 在全历史中的位置越来越分散：

- 连续 PA16 合并机会下降；
- physical page 访问更不规则；
- metadata/page 地址计算占比上升；
- cache/TLB/message stall 可能增加；
- prefill 中 Q3 还要为大量 query 重复完成选中 KV 的读取和 softmax。

当前实测的 head-major + vector metadata 在 PTL 64K/128K prefill 上约带来 1.45× Q3 加速；该结论来自 A/B，不代表已经通过硬件 counters 将 cache、TLB 和消息瓶颈逐项分解。

---

## 8. Prefill 与 decode 的执行差异

| 阶段 | Prefill | Decode |
|---|---|---|
| Q0 | 批量写入当前 query tokens 的 main KV | 写入一个新 token 的 main KV |
| Q1 | 大量 token/block，可使用 DPAS projection 和 block ownership | 主要更新新 token 及可能完成的 compression block |
| Q2 score | query tile × summary tile 的 DPAS | 单 query 的 partition scan |
| Q2 top-k | 大量 score rows，Q2 scratch 需要 row chunking | 单行 score，scratch 很小 |
| Q3 | GQA DPAS，重点是 KV locality 和 metadata | scalar 或 split-selected-KV，重点是并行度 |
| 典型瓶颈 | 长上下文 Q3、Q2 top-k、Q1 projection | Q2 scan/top-k、Q3 split/reduce、固定 dispatch 开销 |

Dense prefix 的意义也不同：

- prefill 短序列中很多 query 的所有 block 都可见，应尽早 bypass score/top-k；
- decode 新 token 若 history 已超过预算，通常仍需完整扫描 summary 选择 512 blocks；
- dense bypass 不能省略 summary 状态更新，也不能把 decode 的历史 selection 假设为固定 block。

---

## 9. 当前性能优化总结

### 9.1 已保留优化

| 优化 | 阶段 | 核心收益 | 语义变化 |
|---|---|---|---|
| DPAS Q1 projection | Q1 prefill | 将 projection 变为 16-token × 8-output tile 的 systolic GEMM | 无 |
| dense score/top-k bypass | Q2 | 对 `C_t<=Kb` 的 query 直接生成 arange | 无，仍维护 Q1 状态 |
| DPAS score tile | Q2 prefill | query tile 和 summary tile 复用，按 indexer head 做 DPAS | 无 |
| bounded row scratch | Q2 | scratch 从按总 query 数分配变为 chunk row 上界 | 无，候选列仍完整 |
| cached fast WG16 | Q2 top-k | keys 一次加载，多轮 scan 复用 | 无，超容量精确回退 |
| head-major scheduling | Q3 prefill | 改善长上下文 KV/page locality | 无 |
| vector metadata gather | Q3 prefill | 减少 selected/page metadata 读取和地址链 | 无 |
| split-selected-KV | Q3 decode | 提高 batch1 decode 的列方向并行度 | 无，使用精确 softmax merge |

### 9.2 已测试但未保留

以下方案没有稳定收益，或增加了资源压力：

- 下一步/两步 KV prefetch：增加 send 和地址计算，反而变慢；
- 显式 cached/uncached/streaming demand hint：没有稳定收益；
- W8：虽然可能减少每线程寄存器，但增加 SLM exchange、softmax 和同步压力；
- step32：减少 barrier 的收益被 register/SLM 压力抵消；
- 只启用 vector metadata：收益小，组合 head-major 后才保留。

### 9.3 解码 top-k 的基线注意事项

decode 的历史结果曾经使用 legacy cooperative-WG16 与 `fast-wg16-cached` 对比。decode 的大幅加速不能全部归因于 key cache，还包含 finalizer 算法选择差异。prefill 的正式 A/B 使用 fast-WG16 与 fast-WG16-cached，比较更直接。

---

## 10. 正确性约束

任何 QSA kernel 优化都必须保持以下语义：

1. **位置语义**：token RoPE 使用 token 绝对位置；summary RoPE 使用 block 起点。
2. **pooling 顺序**：raw K 先按 `r` 个 token 做 FP32 mean，再做 block-key RMSNorm/RoPE。
3. **score 顺序**：每个 indexer head dot → ReLU → head sum → `1/sqrt(Di)`。
4. **top-k 精确性**：保留 finite score 的 ordered key、stable ID tie-break、升序输出和 fallback。
5. **尾部处理**：不完整 compression block 不生成 summary，Q3 直接加入尾部 token。
6. **GQA 映射**：每个 query head 必须访问对应 KV head，不能把 Hq/Hkv ratio 错当成完整 KV 副本数。
7. **softmax 合并**：split partition 必须合并局部 max/sum/weighted value，不能直接平均局部输出。
8. **dense scratch**：dense bypass row 不得读取未写入或旧 chunk 的 score。
9. **paged layout**：逻辑 page 到 physical page 的映射必须通过 block map，不能假设物理连续。
10. **数值边界**：FP32 新 raw K pooling、FP16 cache 回读、FP16 DPAS rounding 形成的舍入边界不能被无意改变。

建议的验证集合包括：空输入、K=0/1/512、可见长度 2047–2053、跨 page、跨 compression block、mixed sequence、padded heads、非完整尾部、全相等 score、signed zero、候选数超过 cache capacity，以及 prefill→decode continuity。

---

## 11. 性能分析与下一步

### 11.1 Roofline 解释边界

PTL 参考峰值为 58 TOPS FP16、110 GB/s；BMG 参考峰值为 118.784 TOPS、456 GB/s。roofline 使用：

$$
 t_{compute}=1000\times\frac{FLOPs}{P},
\qquad
 t_{memory}=1000\times\frac{Bytes}{B},
\qquad
 t_{roofline}=\max(t_{compute},t_{memory}).
$$

`measured/roofline` 只表示实际 kernel 相对于理想化 useful-work 下界的差距，不是硬件利用率，也不是可直接实现的 speedup。Q2 整数比较、SLM、barrier、metadata gather、page/TLB 和 dispatch 开销通常未完整体现在简单 FLOP/byte 模型中。

### 11.2 优先级

1. **Prefill Q1 projection**：继续扩大真正的 DPAS/GEMM 复用，保留 FP32 中间结果。
2. **Dense score/top-k bypass**：短 prefill 可直接省掉 score/top-k，但不能省掉 summary 状态更新。
3. **Q2 cooperative top-k 与 decode score 并行度**：top-k 不是只受 memory bandwidth 限制，还包含 radix/reduction/整数控制。
4. **Decode split-selected-KV**：根据 batch、selected count 和 partition 数寻找 break-even；小请求不应盲目增加 reducer。
5. **Q3 overlap-aware scheduling**：根据 selected-set overlap、连续 page 比例选择 head-major、multi-token tile 或 grouped-head 路径。
6. **硬件 counters**：如果可用，采集 cache/TLB/message/barrier/occupancy，区分 logical bytes 与真实 DRAM traffic。

---

## 12. 代码入口索引

| 功能 | 主要文件 |
|---|---|
| 配置与几何 | `qsa_config.py` |
| Q0 main KV cache | `kernels/qsa_kv_cache_update.cm` |
| Q1 fused prepare | `kernels/qsa_prepare.cm` |
| Q1 DPAS projection | `kernels/qsa_project_dpas.cm` |
| Q2 prefill DPAS score | `kernels/qsa_score_tile_dpas.cm` |
| Q2 decode partition score | `kernels/qsa_score_partition.cm` |
| Q2 top-k finalizer | `kernels/qsa_topk_finalization.cm`, `kernels/qsa_topk_radix_wg.cm` |
| Q3 scalar | `kernels/qsa_sparse_attention.cm` |
| Q3 prefill DPAS | `kernels/qsa_sparse_attention_gqa_dpas.cm` |
| Q3 split/reduce | `kernels/qsa_sparse_attention_split.cm`, `kernels/qsa_sparse_attention_split_reduce.cm` |
| Pipeline wrapper | `test_qsa_pipeline.py`, `qsa_score_chunked.py`, `qsa_attention_gqa.py` |
| PTL roofline | `analyze_ptl_roofline.py`, `PTL_SWEEP_20260911_ROOFLINE_CN.md` |

---

## 13. 总结

QSA 不是一个单 kernel，而是一个带持久化 indexer 状态的稀疏 attention pipeline：

- Q0 保证主 K/V 的 paged cache 正确更新；
- Q1 将当前 hidden 转为 token-level indexer Q、raw K 和 block-level summary；
- Q2 用 summary 估计 block 重要性并精确选择有限 block；
- Q3 使用真实主 K/V 对 selected blocks 执行 causal sparse attention。

当前最关键的优化原则不是盲目扩大向量宽度，而是减少不必要工作、提高正确的并行粒度，并尊重 page、compression、GQA、RoPE、top-k tie-break 和 softmax 合并语义。prefill 应优先处理 projection、dense bypass 和长上下文 Q3 locality；decode 应优先处理 summary score/top-k 的并行度和 split-selected-KV 的 reducer 成本。
