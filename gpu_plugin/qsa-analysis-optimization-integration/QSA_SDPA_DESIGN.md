# QSASDPA 设计:让 QSA 同时支持 SDPA 风格的对外接口

> **实现状态(2026-09-16)**:阶段 0-2(退化分页验证、`skip_kv_write`、`QSASDPA`/
> `QSAIndexerCache` 两个 op 本体)均已完成。**阶段 3 已全部完成**:
> `ops/qsa_sdpa.cpp::CreateQSASDPAOp` 与 `ops/qsa_indexer_cache.cpp::
> CreateQSAIndexerCacheOp` 两个 ProgramBuilder lowering 均已实现,并且——不同于
> 阶段 3a 首次交付时的范围——**`QSAIndexerCache` 现在真正通过 `ReadValue → Q1
> (prepare_only) → Assign` 在两次独立的 `ov::InferRequest::infer()` 调用之间
> 正确持久化 indexer raw-K/summary 状态**,已用一个显式跨两次真实 GPU 调用的
> 端到端测试验证(见文末"实现与验证记录")——这正是之前"QSASDPA 端到端测试"里
> 明确指出未覆盖的那个缺口。仍未实现:`QSASDPAFusion` 图匹配 pass、beam 重排
> (按用户要求本轮不实现)。请先读文末记录再继续。这一步发现并澄清了一个此前设计
> 未明确的关键约束:**`pa_block_size`
> 是烘焙进 CM kernel 的 JIT 编译期常量,因此退化分页的"页大小"(即批内共享的最大
> 序列长度容量)和批内序列数 `B` 都必须在图构建期静态已知**——`CreateQSASDPAOp`
> 已将这一点实现为显式校验(拒绝动态 `B`/动态 `pa_block_size`),细节见 4.4 节。
> `QSAIndexerCache` 自己的 ProgramBuilder lowering(含 `Variable` 状态持久化)与
> `QSASDPAFusion` 图匹配 pass 仍**未实现**,见文末记录的详细分析和后续计划。Indexer
> raw-K/summary 的状态管理方案已在 2026-09-16 定案为**方案二**(见 4.3 节)。
> beam 重排(`QSABeamReorderOpt`)按用户要求本轮**明确不实现**,不在当前范围内。

## 1. 文档目的与范围

本文设计在 `openvino.mx` Intel GPU plugin 中新增一个对外算子 `QSASDPA`,让 QSA(Qwen
Sparse Attention)在**不改变现有 `QSA`(PagedAttention 风格)算子和 kernel 的前提下**,
也能以 `SDPA`(`ov::op::v13::ScaledDotProductAttention` / `ov::intel_gpu::op::SDPA`)
那种"稠密 Q/K/V + `KVCache`/`Variable` 状态管理"的方式被外部调用方使用。

对应三条需求:

1. 新增 `QSASDPA` 算子,外部调用方不需要理解分页/页表概念即可使用 QSA。
2. 尽可能复用现有 PagedAttention 风格的 QSA kernel(Q0-Q3 全部 `.cm` 文件、`qsa.cpp`
   里的所有 `KernelGenerator`),优先采用"退化分页"方案落地,`QSA` 和 `QSASDPA` 两个
   算子互不影响、可以在同一个 plugin 里同时存在。
3. `QSASDPA` 的 KV cache 生命周期管理方式与 `SDPA` 保持一致,即基于
   `ov::op::util::Variable` + `ReadValue`/`KVCache` 融合机制,而不是像 `QSA` 那样把
   物理页表和缓存张量当作外部调用方直接传入的 graph input。

### 1.1 与既有设计原则的关系(先澄清,避免误读)

`OV_SUPPORT_QSA_GR_DESIGN_CN.md` 第 2 节的约束 3 写的是"不复用 `paged_attention` 或
`scaled_dot_product_attention` 的 **primitive、graph implementation 或 kernel**;
它们只能作为接口风格参考"。本设计**不违反**这条约束:

- `QSASDPA` 复用的是 **QSA 自己的** kernel(`qsa_kv_cache_update.cm`/`qsa_prepare.cm`/
  `qsa_score_*.cm`/`qsa_sparse_attention_*.cm`)和 **QSA 自己的** `cldnn::qsa`
  primitive——不是 `paged_attention` 或 `scaled_dot_product_attention` 的 kernel。
- 唯一跨界复用的是 `ov::intel_gpu::op::KVCache` / `IndirectKVCache`——这是一个**通用的
  状态管理基础设施**,`SDPA`/`IndirectSDPA`/`Gemm`/`IndirectGemm` 都只是它的"消费者",
  它本身不属于"`paged_attention` 或 `scaled_dot_product_attention` 的实现";复用它
  管理状态,和"复用 SDPA 的 attention 计算逻辑"是两回事,不构成对约束 3 的违反。

## 2. 现状回顾(摘要,细节见前序分析)

### 2.1 `QSA`(PagedAttention 风格)现状

`ov::op::internal::QSA` / `ov::intel_gpu::op::QSA` 的 17 个输入([qsa.hpp](../openvino.mx/src/core/dev_api/openvino/op/qsa.hpp))里,`PAST_LENS`/`SUBSEQUENCE_BEGINS`/
`BLOCK_INDICES`/`BLOCK_INDICES_BEGINS`/`KEY_CACHE`/`VALUE_CACHE`/
`INDEXER_RAW_K_CACHE`/`INDEXER_SUMMARY_CACHE` 8 个输入是纯 PagedAttention 概念:
物理页表由外部调用方(vLLM 式 continuous-batching scheduler)管理,作为普通 graph
input 每次调用传入,cldnn primitive 自己不持有任何 `Variable`/状态。

Q0-Q3 四个 kernel 家族的寻址公式(`qsa::kv_cache_offset`/`raw_k_offset`/
`summary_offset`,见 [qsa_common.hpp](../openvino.mx/src/plugins/intel_gpu/src/graph/impls/cm/include/qsa_common.hpp))全部是"两级寻址":
`block_indices[逻辑 block id]` 查出物理 page,再在 page 内按 `pos % PA_BLOCK_SIZE`
偏移。host 端 `qsa.cpp::update_rt_params()` 的 `QSAStage`/`prepare_map`/
`score_tile_map`/`token_map` 全部基于 `past_lens`/`subsequence_begins` 这种 ragged
token 展开来构造。

因果(causal)语义是结构性的,与分页/稠密无关:`qsa_num_complete_blocks(abs_pos) =
(abs_pos+1)/R` 决定哪些 block "完整可见"从而进入 top-k 候选池,Q3 额外处理一个
"当前不完整 block 的因果尾部"(`tail_begin`/`tail_rows`)。**这套逻辑在 QSASDPA 里
原样复用,不需要改一行。**

### 2.2 "退化分页"方案的可行性结论(已验证)

把 `pa_block_size` 设成 `max_seq_len`、`block_indices` 设成恒等映射(page i ↔ 序列
i),数值上完全正确(causal/寻址公式都不关心 page 大小),但有两个已定量的代价:

- **内存**:4 个持久 cache 必须按 `batch × max_seq_len` 全额预留,不能像真实分页那样
  按实际用量增量分配。
- **`Q2` scratch/chunk 数**:`get_internal_buffer_descs()`/`score_scratch_rows()`
  里 `max_blocks = kcache[0] * pa_block_size / compress_ratio`
  ([qsa.cpp](../openvino.mx/src/plugins/intel_gpu/src/graph/impls/cm/qsa/qsa.cpp#L132)),退化方案下 `kcache[0]`(物理 page 数)= batch 大小,导致这个上界
  随 batch 线性膨胀,`scratch_rows`(每个 Q2 chunk 能装的 query 行数)成比例变小,
  chunk 数变多。运行时 dispatch 网格(`update_rt_params()` 里真正用的 `max_blocks`)
  不受影响,只有分配时刻的静态估算受影响。

结论:退化分页在 **batch 较小/单序列服务场景下近乎零代价**,batch 较大时需要按本节
上面的公式实际核算——这是本设计默认选择这条路径、但同时在第 4.8 节保留缓解手段的
原因。

### 2.3 `SDPA` 的 KV cache 管理机制(即 QSASDPA 要对齐的目标)

标准 OV 有状态模型契约:`ReadValue(Variable)` → `(Gather beam_idx)` →
`Concat(past, new_token)` → 一份喂 `SDPA`,一份 `Assign(Variable)`。GPU 插件的
`KVCacheFusion`([kv_cache_fusion.cpp](../openvino.mx/src/plugins/intel_gpu/src/plugin/transformations/kv_cache_fusion.cpp))把这个模式重写成
`ov::intel_gpu::op::KVCache`(`VariableExtension`,融合了"拼接"和"状态写回"两个职责,
可以做原地追加式增长)+ 专用 `ReadValue`。`IndirectKVCache`/`IndirectGemmOpt`/
`IndirectSDPAOpt`([indirect_kv_cache.cpp](../openvino.mx/src/plugins/intel_gpu/src/plugin/transformations/indirect_kv_cache.cpp))进一步把 beam search 的 `Gather` 融合进
`KVCache`(多出一个 `beam_table` 输出),下游换成 `IndirectSDPA`/`IndirectGemm` 在
读取时才做逐行 gather,避免物理搬运整份 cache。`KVCache` 还支持 `trim`(滑窗淘汰)和
`update_kv`(树形推测解码原地覆写)。

`SDPA`/`IndirectSDPA` 本身**不持有任何状态**:它拿到的 `KEY`/`VALUE` 就是
`KVCache` 这一步产出的、已经包含全部历史的稠密张量,causal 语义是对这个全量张量重新
算一遍 `is_causal`/`attn_mask`。

## 3. 总体设计

### 3.1 核心决策

| 决策点 | 选择 | 理由 |
|---|---|---|
| 新 kernel 还是复用? | **复用**——`QSASDPA` 不新增任何 `.cm` kernel,Q0-Q3 全部沿用 `qsa.cpp` 现有的 `KernelGenerator`/`Stage` | 满足需求 2;避免重新验证一遍 dpas 数值正确性 |
| 分页寻址还是稠密寻址? | **退化分页**(`pa_block_size=max_seq_len`,恒等 `block_indices`),作为 lowering 阶段在 `ProgramBuilder` 内部合成,不暴露给模型作者 | 满足需求 2 的"优先退化分页";已在第 2.2 节验证数值正确、代价可核算 |
| 新 op 还是给 `QSA` 加 mode? | **新增独立 op `QSASDPA`**,不改 `QSA` 的 `Config`/`InputIdx` | 满足需求 2 的"互不影响";镜像 `SDPA`/`PagedAttention` 分立、`IndirectSDPA : SDPA` 的既有先例 |
| 主 K/V cache 归属 | 外部 `ov::intel_gpu::op::KVCache`(**原样复用**,零新增),`QSASDPA` 拿到的是已增长好的稠密张量,和 `SDPA` 完全一致 | 满足需求 3;顺带让 Q0(`qsa_kv_cache_update.cm`)在 `QSASDPA` 路径下**不需要执行**——追加已经由 `KVCache` 做了 |
| Indexer raw-K + summary cache 归属(**方案二,2026-09-16 定案**) | **新增**一个算子 `ov::intel_gpu::op::QSAIndexerCache`(`VariableExtension`),内部**原样复用 Q1(`qsa_prepare.cm`)kernel**,只是用退化分页(恒等 `block_indices`)代替物理页表寻址——不外包给通用 `KVCache` | Q1 把"indexer K 投影"和"raw-K 追加"融合在同一个 kernel 里,raw-K 不是可以直接喂给通用 Concat 的现成张量(方案一的前提不成立,见 4.2/4.3 节);方案二零新增 kernel、复用度最高、风险最低 |
| 每次调用各序列新增 token 数是否可以不同? | **可以**,保留 `PAST_LENS`/`SUBSEQUENCE_BEGINS` 作为 `QSASDPA` 的显式输入(而不是简化成单个 `SEQ_LENS`),复用 `QSA` 已有的 `QSAStage::MIXED` 分派 | 满足新增需求 1;host 端 `update_rt_params()` 的 `MIXED` 判定逻辑对 `QSASDPA`/`QSA` 完全共用,零新增分派代码 |
| Beam search 下 raw-K/summary 是否重排? | **重排**,但 v1 用"及早物化"(显式 `Gather`)而不是扩展 QSA kernel 支持 `beam_table` 间接寻址 | 满足新增需求 2;避免修改 Q1/Q2/Q3 kernel,保持 3.1 节"零新增/零修改 kernel"的核心目标 |
| 运行时可否动态增删 batch 内序列(动态 `B`)? | **v1 不支持**,视为固定,见第 7 节 | 退化分页的恒等页表前提依赖 `B` 在一次编译生命周期内不变;支持它等价于放弃"退化"这个简化,应直接使用 `QSA`(分页模式) |

### 3.2 端到端数据流

```mermaid
flowchart TB
    subgraph 模型图(调用方视角,不出现分页概念)
        MIXED[MIXED 隐藏态] --> QSDPA[QSASDPA]
        MIXED --> IdxCache[QSAIndexerCache]
        K_new[新 token 的 Key] --> KVC_K[KVCache: 主 Key]
        V_new[新 token 的 Value] --> KVC_V[KVCache: 主 Value]
        KVC_K --> QSDPA
        KVC_V --> QSDPA
        IdxCache -->|raw-K dense + summary dense| QSDPA
    end
    QSDPA -->|ProgramBuilder lowering| LowerA[cldnn::qsa, skip_kv_write=true<br/>Q1+Q2+Q3 原样执行]
    IdxCache -->|ProgramBuilder lowering| LowerB[cldnn::qsa, prepare_only=true<br/>仅 Q1 原样执行]
    LowerA --> Out[attention 输出]
```

`KVCache: 主 Key/Value` 是**对已有 `ov::intel_gpu::op::KVCache` 的直接实例化**,不
新增任何 C++ 代码。`QSAIndexerCache` 是本设计唯一需要新写的算子,但它**执行的内容是
Q1(`qsa_prepare.cm`)kernel 本身,一行不改**——只是通过 `cldnn::qsa` 新增的
`prepare_only` 标志(已实现并在 BMG 上验证,见文末"实现与验证记录")只跑 Q1、跳过
Q0/Q2/Q3,寻址方式和 `QSASDPA` 主体侧完全一样,都是恒等 `block_indices` 的退化
分页。

**已知的权衡(方案二带来的,不是 bug)**:`QSASDPA` 自己的 `cldnn::qsa` 实例(见
LowerA)仍然会**原样跑一遍 Q1**(`skip_kv_write` 只跳过 Q0,不跳过 Q1)——因为 Q2
需要 Q1 当场算出的、本次调用的 indexer Q(RMSNorm+RoPE 之后的),而这个值不持久化、
不是 `QSAIndexerCache` 的输出。`QSASDPA` 侧的 raw-K/summary 输入是 `QSAIndexerCache`
已经算好的稠密快照(普通数据依赖,不是它自己的 `Variable`),`QSASDPA` 的 Q1 会把
这份快照**原样再写一遍**(得到完全相同的值,幂等、无害,只是浪费算力)。也就是说
**Q1 这次会跑两遍**(一遍在 `QSAIndexerCache` 里维护持久状态,一遍在 `QSASDPA`
里算当次 indexer Q + 局部 scratch)。这是方案二为了不碰 Q1 kernel、不引入第二个
Variable 所有者而接受的确定性代价,量级明确(至多 2 倍 Q1 开销),不是正确性问题;
真要消除它需要真正拆开 Q1(方案一的路子),留作未来可选优化,不在本次范围内。

## 4. 详细设计

### 4.1 新 op:`ov::intel_gpu::op::QSASDPA`

镜像 `ov::intel_gpu::op::QSA` 但去掉 6 个纯物理分页专属输入(`BLOCK_INDICES`/
`BLOCK_INDICES_BEGINS`/`KEY_CACHE`/`VALUE_CACHE`/`INDEXER_RAW_K_CACHE`/
`INDEXER_SUMMARY_CACHE`),`InputIdx` 改为:

```
MIXED, QUERY,                              // 与 QSA 相同,仍是按 token 展平的 ragged 张量
PAST_LENS,                                 // [num_seqs],与 QSA 完全相同语义:每序列本次调用前的历史长度
SUBSEQUENCE_BEGINS,                        // [num_seqs+1],与 QSA 完全相同语义:QUERY 的每序列偏移
KEY_CACHE_DENSE,  VALUE_CACHE_DENSE,        // 来自外部 KVCache 节点,已含全部历史(含 padding,见下)
INDEXER_RAW_K_CACHE_DENSE,                  // 来自 QSAIndexerCache 的输出0(见 4.3 节)
INDEXER_SUMMARY_CACHE_DENSE,                // 来自 QSAIndexerCache 的输出1
INDEXER_QK_WEIGHT, INDEXER_Q_NORM_WEIGHT,
INDEXER_K_NORM_WEIGHT, ROTARY_COS_SIN,      // 与 QSA 完全相同,原样透传
POSITION_IDS                                // 与 QSA 相同(仍需要,决定 abs_pos/RoPE)
```

**保留 `PAST_LENS`/`SUBSEQUENCE_BEGINS` 而不是简化成单个 `SEQ_LENS`,是为了满足
"一次调用里不同序列可以新增不同数量 token"这一需求**(等价于 `QSA` 分页模式的
`QSAStage::MIXED`:同一个 batch 里既有只新增 1 个 token 的 decode 序列,也有新增多个
token 的 prefill/续写序列)。`QUERY` 因此和今天的 `QSA` 一样是**展平(ragged)**张量,
不是一个干净的 `[B, T_new, ...]` 稠密张量;`SUBSEQUENCE_BEGINS` 负责标出每个序列在
这个展平张量里的起止位置,`PAST_LENS` 负责标出每个序列调用前的历史长度。

`KEY_CACHE_DENSE` 等四个持久态输入是**稠密但可能含 padding** 的张量:`KVCache`
状态的 `S` 轴是整个 batch **共享**的一个维度(取 batch 内最长历史的长度),历史更短的
序列在这个轴上是 padding,而不是"没有这一行"。`QSASDPA`/底层 `cldnn::qsa` 完全不需要
关心这个 padding——因果与 top-k 逻辑只根据 `PAST_LENS[s]` 判断每个序列真正可见到
哪里,padding 部分自然落在 `abs_pos` 之外不会被访问,和真实分页模式下"某些逻辑
block 干脆不存在"是等价的边界处理,不需要新写 mask。

`Config` 与 `ov::op::internal::QSA::Config` 完全相同字段(`heads_num`/`kv_heads_num`/
`compress_ratio`/`budget`/`is_causal`/…),`validate_config()` 直接复用——因为底层
lowering 到的还是同一个 `cldnn::qsa` primitive 的 `Config`,几何约束没有变化。

### 4.2 主 Key/Value:直接复用 `KVCache`

主 Key/Value 由外部常规 `k_proj`/`v_proj` 计算(QSA 从不做这个投影),新增值是一个
现成的稠密张量,和标准 SDPA 场景的 K/V cache 增长语义**完全同构**——模型图里对应
的写法就是标准 `ReadValue` + `Concat(axis=token)` + `Assign`,`KVCacheFusion` pass
会**自动**把它识别并重写成 `ov::intel_gpu::op::KVCache`,不需要为 QSA 写任何新的
图匹配逻辑。

需要新增的只是 `QSASDPAFusion`(见 4.5 节)要能正确识别"`QSASDPA` 消费的是某个
`KVCache` 节点的输出",以便 lowering 阶段拿到该 `KVCache` 的 `Variable`(从而知道
当前 batch/seq_len,用于合成退化分页元数据)。

**Indexer raw-K/summary 不适用这个模式**——原方案曾把 raw-K 也归入这一类,实现
阶段发现前提不成立(Q1 把 indexer K 投影和 raw-K 追加融合在同一个 kernel 里,
raw-K 不是可以直接喂给通用 Concat 的现成张量),已改为 4.3 节的方案二。

### 4.3 Indexer raw-K + summary:`QSAIndexerCache`(方案二,已定案并验证)

**候选方案回顾**:方案一是把 indexer K 投影从 Q1 里拆出来,变成图上一个独立
MatMul,再用通用 `KVCache` 追加 raw-K,summary 再配一个新的小 kernel;方案二是保持
Q1 融合不变,给 raw-K/summary 状态包一个专属的 `VariableExtension` 算子。方案一能
让 raw-K 也享受通用 `KVCache`(以及它免费的 beam 重排),但需要新写 kernel、拆分
Q1 带来数值风险、并让 `QSA`(分页模式)和 `QSASDPA` 的 indexer 端变成两套代码。方案
二零新增 kernel、和已验证的 Q1 数值风险最低、外部调用方需要理解的图节点最少、与
退化分页这套已验证的心智模型完全一致。**结论:采用方案二**(如果 beam search 支持
的成本、或"raw-K 也要字面通用 KVCache"是硬性要求,可退回方案一的折中——只拆
raw-K,summary 仍走专属小 kernel,本设计不默认采用)。

```cpp
// v1 已实现(见文末"实现与验证记录"),不含 beam_idx/indirect 模式(见 4.9 节,
// 留给后续阶段)。VariableExtension 接口只支持单个 Variable,raw-K 走
// get_variable()/get_variable_id()(VariableExtension 契约),summary 通过新增的
// get_summary_variable()/get_summary_variable_id() 显式访问器暴露——下游任何需要
// 同时处理两份状态的代码(状态重置、beam 重排、ProgramBuilder lowering)都必须用
// 这组显式接口,不能假设 VariableExtension 能拿到全部状态。
class QSAIndexerCache : public ov::op::Op, public ov::op::util::VariableExtension {
public:
    OPENVINO_OP("QSAIndexerCache", "gpu_opset");
    using Config = ov::op::internal::QSA::Config;
    // inputs: MIXED(隐藏态,与 QSASDPA 共享同一份图输出)、INDEXER_QK_WEIGHT、
    //         INDEXER_Q_NORM_WEIGHT(Q1 内部计算 Q 时用到,但 Q 本身不持久化)、
    //         INDEXER_K_NORM_WEIGHT、ROTARY_COS_SIN、PAST_LENS、SUBSEQUENCE_BEGINS、
    //         POSITION_IDS(Q1 的 RoPE 需要绝对位置,一般无法从 past_lens 推出,
    //         mRoPE 场景尤其不行——实现阶段发现的、原设计遗漏的一个真实输入,不是
    //         死值,已补进 InputIdx,v1 共 8 个;可选 beam_idx 留给 beam search
    //         场景,见 4.9 节,v1 未实现)
    // output0: raw_k_dense   [B, S, Di]           (present,逐 token 追加)
    // output1: summary_dense [B, ceil(S/R), Di]   (present,每 R 个 token 追加一个)
    // output2(仅未来 indirect 模式): beam_table

    std::string get_variable_id() const override;               // -> raw-K variable's id
    std::shared_ptr<ov::op::util::Variable> get_summary_variable() const;
    std::string get_summary_variable_id() const;
};
```


**执行内容**(已实现并在真实 BMG GPU 上验证,见文末"实现与验证记录"):
`QSAIndexerCache` lowering 到的还是同一个 `cldnn::qsa` primitive,只是新增了一个
`prepare_only` 标志——为真时只 `add_stage` Q1(`qsa_prepare`/`qsa_project_dpas`),
Q0/Q2/Q3 全部跳过,`execute()` 在 Q1 完成后直接返回。寻址方式和 `QSASDPA` 主体侧
完全一样,都是恒等 `block_indices` 的退化分页。Q1 kernel 本身
(`qsa_prepare.cm`)**一行不改**:它已经知道如何从任意的 `past_len` 状态开始,只
处理"这次新增的、够格产生新 raw-K/summary 条目"的那一小段——这正是增量计算的核心
优势,不会退化成"每步重算全部历史"。

`trim`/`reset_state()` 语义与 `KVCache` 保持一致(精神一致,非字面同一个 op 类):
`ov::InferRequest::reset_state()` 清空这里的 `Variable` 和主 K/V 的 `KVCache`
`Variable`,效果等价于开始一个全新序列。Beam search 场景下的重排需求(每一步都要
用同一个 `beam_idx` 同时重排主 K/V、raw-K、summary 三份状态)在第 4.9 节统一说明。

**已知的权衡**(见 3.2 节末尾):`QSASDPA` 自己的 `cldnn::qsa` 实例仍需要原样跑一遍
Q1 来算当次 indexer Q(Q2 要用,且不持久化),这意味着 Q1 这次调用里跑了两遍——
一遍在 `QSAIndexerCache` 里维护持久状态,一遍在 `QSASDPA` 里重算(幂等、结果相同,
只是浪费算力)。这是接受方案二"零新增/零修改 kernel"的确定性代价,量级可控
(至多 2 倍 Q1 开销),真要消除需要真正拆开 Q1(方案一的路子),留作未来可选优化。

### 4.4 ProgramBuilder lowering:合成退化分页元数据(已实现,见文末记录)

新增 `ops/qsa_sdpa.cpp::CreateQSASDPAOp`,在图构建期(不是每次 `execute()`)做:

**关键澄清(实现时发现,原设计未明确)**:`pa_block_size` 不是一个运行时可变的值——
它被 `qsa.cpp`(CM 实现)通过 `jit.make("QSA_PA_BLOCK_SIZE", desc->pa_block_size)`
**烘焙成 CM kernel 的 JIT 编译期常量**,对每个 kv_head 的地址跨度(`physical_page *
pa_block_size * head_size`)起决定性作用。这意味着退化分页"一个序列一页,页大小
等于当前总长度"这句话必须精确为:**`pa_block_size` 是在图构建期(`compile_model`
时)一次性选定的固定容量上限**(设计初稿里"等于当前 S"的说法只在"预先按上限
`pa_block_size` 一次性分配好、之后每次调用都在同一容量内工作"这个前提下成立,
不是"每次调用都换一个不同的 `pa_block_size` 重新编译"),`B`(批内序列数)同理必须
在图构建期静态已知——这正是设计里"v1 不支持动态 B"这条限制的**具体、可执行**
形式,而不只是一句手写声明。`CreateQSASDPAOp` 把这一点实现成显式的
`OPENVINO_ASSERT`:`KEY_CACHE_DENSE` 的 `PartialShape` 必须完全静态,其 `S` 轴
必须等于 `config.pa_block_size`,`PAST_LENS` 的批维度必须静态且等于同一个 `B`——
不满足时在 `compile_model()` 阶段就以清晰的错误信息拒绝,不会静默产生错误数值
(已有专门测试验证,见文末记录)。

在这个"`B`/`pa_block_size` 均静态已知"的前提下,lowering 本身变得比原设计设想的
简单得多,不需要任何 `ShapeOf`/`Range`/`Concat` 之类的动态图手术:

1. 从 `KEY_CACHE_DENSE` 的静态 `PartialShape` 直接读出 `B`(`shape[0]`)和
   `S`(`shape[2]`,即 `config.pa_block_size`),连同 `Config` 里的
   `Hkv`/`Dh`/`Dv`/`Di`/`compress_ratio`,是纯粹的编译期 C++ 常量运算。
2. `KEY_CACHE_DENSE`/`VALUE_CACHE_DENSE` 已经是 `[B, Hkv, S, Dh/Dv]`——与
   `QSA::InputIdx::KEY_CACHE`/`VALUE_CACHE` 期望的 `[num_blocks, Hkv, PA_BLOCK_SIZE,
   Dh/Dv]` **逐维度相同**(退化分页下 `num_blocks == B`),原样透传,不需要任何
   reshape。
3. `INDEXER_RAW_K_CACHE_DENSE`/`INDEXER_SUMMARY_CACHE_DENSE` 是 `[B, S, Di]`/
   `[B, S/r, Di]`(3 维,QSASDPA 对外的"稠密"约定),而
   `QSA::InputIdx::INDEXER_RAW_K_CACHE`/`INDEXER_SUMMARY_CACHE` 需要 `[B, 1, S,
   Di]`/`[B, 1, S/r, Di]`(4 维,多一个恒为 1 的 head 轴)——用一个纯重新标注、不
   搬运数据的 `cldnn::reshape`(静态 `output_pattern`)插入这个 size-1 维度。
4. 合成 2 个恒等映射常量(`cldnn::data`,在 lowering 时用 host 端 C++ 循环直接算好
   数值,不是运行时算子):`block_indices = [0, 1, ..., B-1]`、
   `block_indices_begins = [0, 1, ..., B]`。
5. 合成 2 个哑元零常量(`cldnn::data`,极小 shape、内容全零),喂给底层
   `cldnn::qsa` 的 `QSA::InputIdx::KEY`/`VALUE`——这两个槽位只有 Q0
   (`qsa_kv_cache_update`)会读,而 `skip_kv_write=true` 时 Q0 永远不会被
   `add_stage()`,所以它们的具体内容完全不重要,只需要满足"存在一个类型/rank
   合法的输入"这个 C++ 层面的构造要求。
6. `past_lens`/`subsequence_begins` **不需要额外合成**——`QSASDPA` 的
   `PAST_LENS`/`SUBSEQUENCE_BEGINS` 输入(见 4.1 节)本身就是 `cldnn::qsa`
   期望的确切格式,原样透传即可。这也是 `MIXED`(见 4.1 节)几乎不增加 lowering
   复杂度的原因:host 端 `QSAStage::PREFILL/GENERATE/MIXED` 判定、
   `prepare_map`/`score_tile_map`/`token_map` 构造(`update_rt_params()`)对
   `QSASDPA` 和 `QSA` 走的是同一份代码,不需要区分调用来源(已有 MIXED 端到端
   测试验证,见文末记录)。
7. 构造一个 **完整复用** 现有 `qsa.cpp` 里 `KernelGenerator`/`Stage` 派发路径的
   `cldnn::qsa` primitive 实例(`skip_kv_write=true`,`prepare_only=false`)——即
   `QSASDPA` 最终在 cldnn 图里就是一个 `cldnn::qsa` 节点,Q0 阶段被跳过(见
   4.5),Q1/Q2/Q3 阶段与今天的 `QSA` 走的是**同一份** `KernelGenerator`/`Stage`/
   JIT 常量代码,不存在第二套 kernel。

**尚未覆盖的场景(下一步)**:同批内序列历史长度不同(需要"共享 S 轴 + 短序列
padding"逻辑,MIXED 端到端测试目前只验证了"批内序列 `total_len` 相同"的情形)、
一次编译内的 `trim`/滑窗淘汰(第 4.8 节的可选缓解手段)。这两者都不影响本节已实现
的核心 lowering 逻辑,是它之上的增量特性。

### 4.5 Q0 在 `QSASDPA` 路径下不需要执行

`qsa_kv_cache_update.cm`(Q0)的职责是"把当前新 token 的 K/V 写入分页 cache"。在
`QSASDPA` 路径下,这件事已经被外部 `KVCache` 节点在 `QSASDPA` 执行**之前**做完了
(`KVCache` 的 `Concat` 语义本身就是追加)。因此:

- `QSACMImplementationManager` 需要一个新的运行期分支(不是新 primitive 类型,是
  同一个 impl 内部按 `desc->skip_kv_write`——由 lowering 阶段设置的一个新
  `bool` 字段——决定是否 `add_stage()` Q0),`QSA`(分页模式)路径该字段恒为
  `false`,`QSASDPA` 路径恒为 `true`。
- 这是本设计里**唯一**需要在 `cldnn::qsa`/`qsa.hpp`/`qsa.cpp` 里加的新字段,且是一
  个纯粹的"跳过某个 stage"开关,不影响 `QSA`(分页模式)路径的任何既有行为
  ——满足需求 2 的"互不影响"。

### 4.6 与既有 `QSA`(分页模式)的隔离保证

| 层面 | 隔离方式 |
|---|---|
| OpenVINO op | `QSASDPA` 是独立类,不修改 `ov::op::internal::QSA`/`ov::intel_gpu::op::QSA` 的 `Config`/`InputIdx`/`validate_and_infer_types` |
| cldnn primitive | 仍然只有一个 `cldnn::qsa` primitive 类;新增的唯一字段 `skip_kv_write` 默认 `false`,`QSA`(分页模式)lowering 路径不设置它,行为逐字节不变 |
| kernel | 零新增/零修改 `.cm` 文件;Q0 的 `add_stage()` 调用外面包一层 `if (!desc->skip_kv_write)`,`false` 时该 `if` 恒真,原有代码路径不变 |
| factory 注册 | 新增独立的 `ops/qsa_sdpa.cpp::CreateQSASDPAOp` + `REGISTER_FACTORY_IMPL(internal, QSASDPA)`,与 `ops/qsa.cpp::CreateQSAOp` 互不干扰(参考仓库既有 gotcha:两者必须各自注册,不能共用 `using X = ...` 别名技巧,否则会重演此前 `QSA`/`GatedResidual` 暴露 dev_api 时踩过的 "conflicting declaration" 编译错误) |
| 测试 | `qsa_gpu_test.cpp` 新增独立的 `qsa_sdpa_gpu_test.cpp`,不修改任何现有 `TEST`/`TEST_P` |

### 4.7 因果掩码语义:完全不变

`qsa_num_complete_blocks`/tail 逻辑只依赖 `abs_pos`(由 `past_lens`+
`subsequence_begins` 算出),与"这两个输入是调用方直接给的"还是"host 端从
`SEQ_LENS` 派生的"无关。**Q1/Q2/Q3 三个阶段的因果正确性证明在第一轮分析里已经完成,
这里不需要重新证明,只需要保证第 4.4 节合成出的 `past_lens`/`subsequence_begins`
数值语义与今天 `QSA`(分页模式)的调用方约定完全一致即可。**

### 4.8 效率代价与缓解(沿用第 2.2 节的量化结论)

- **默认接受**:退化分页在 batch 较小的服务场景(单流或小并发)下代价可忽略,这是
  绝大多数"不用 vLLM continuous batching、只是想直接调 QSA"场景的真实画像。
- **大 batch 场景的缓解手段**(按需实现,不在首个交付范围内):
  1. 在 `CreateQSASDPAOp` 里允许配置一个"预算 `max_seq_len_hint`"上限,而不是让
     `pa_block_size` 跟着运行时最大 `S` 无限增长,超过上限时触发 `KVCache` 的
     `trim` 语义(滑窗淘汰最老的历史),从根上限制退化分页的膨胀系数。
  2. 观察到 `max_blocks` 膨胀的根因是"`kcache[0]` 复用了物理 page 数"这一个公式
     (`qsa.cpp` 第 132/1195 行),如果未来需要更精细的控制,可以在 `skip_kv_write`
     之外再加一个 `dense_batch_hint` 字段,让 `score_scratch_rows()` 按
     "`S/compress_ratio`"而不是"`kcache[0]*S/compress_ratio`"估算——这是一处
     **可选的** host 端公式修正,不涉及 kernel,风险很低,可以作为二期优化。

### 4.9 Beam search 下 raw-K/summary cache 的重排

Beam search 每一步都要用同一个 `beam_idx` **同时、一致地**重排四份持久状态:主
Key、主 Value、indexer raw-K、indexer summary——四份状态对应同一组 beam 假设,任何
一份没跟着重排,就会出现"新 token 的主 K/V 来自 beam A,但 summary 历史来自 beam
B"的不一致。

**v1 方案:复用 `IndirectKVCache` 的检测机制,但选择"及早物化"(eager
materialize)而不是"惰性间接读取"(true indirect),取舍如下:**

- 主 K/V 两份状态本身就是普通 `ov::intel_gpu::op::KVCache` 实例,`
  IndirectKVCache`/`IndirectGemmOpt`/`IndirectSDPAOpt` 这套现有 pass 本来就会
  识别它们前面的 `Gather(beam_idx)` 并融合进 `KVCache`(`m_indirect=true` +
  `beam_table` 输出)——这一步不需要新代码,是既有机制的自然复用。
- 但现有 QSA 的 Q1/Q2/Q3 kernel(`qsa_prepare.cm`/`qsa_score_*.cm`/
  `qsa_sparse_attention_*.cm`)**完全不认识** `beam_table` 这个概念,它们的寻址
  只理解 `block_indices` 这一种页表间接层,没有第二层"按 beam 再间接一次"的能力
  ——要让 QSA kernel 像 `IndirectSDPA` 那样在读取的那一刻按 `beam_table` 做逐行
  gather,需要修改 Q1/Q2/Q3 的寻址代码,这违背了 3.1 节"零新增/零修改 `.cm`
  kernel"的核心目标。
- 因此 v1 新增一个小的图变换 pass `QSABeamReorderOpt`:检测到某个 `KVCache`(已被
  `IndirectKVCache` 转成 indirect 模式、产出 `beam_table`)被 `QSASDPA` 或
  `QSAIndexerCache` 消费时,**插入一个显式的 `Gather(dense_tensor,
  beam_table, axis=batch)`**,把重排后的结果再喂给 `QSASDPA`/
  `QSAIndexerCache`——这等价于 `IndirectKVCache` 优化**之前**、`SDPA` 场景
  下最早期的朴素做法(显式 `Gather` 整份 cache),正确性没有任何疑问,代价是每个
  decode 步骤多一次 `O(B×S×D)` 的显式搬运,而不是 `IndirectSDPA` 那种零搬运的
  惰性 gather。
- `QSAIndexerCache` 自身(4.3 节的 `m_indirect`/`beam_table` 字段)
  用同样的方式处理:它的 `Variable`(s) 存的是已经产出的 raw-K/summary 历史,beam
  重排同样要作用在这份历史上,消费的是同一个 `beam_table`,插入同一种显式 `Gather`。

**未来可选优化(不在 v1 范围内)**:如果显式 `Gather` 的搬运开销在实测中成为
瓶颈,可以再引入"真正的间接 QSA kernel"——让 Q1/Q2/Q3 的 host 端 dispatch 额外
接受一个 `beam_table` 参数,在 `qsa::kv_cache_offset`/`raw_k_offset`/
`summary_offset` 之外再做一层 `physical_seq = beam_table[logical_seq]` 的间接。
这是一处真实的 kernel 改动,需要重新走一遍数值验证,因此明确列为二期工作,不放进
首个交付范围。

## 5. 阶段化实施计划

1. **阶段 0(验证性原型,不动生产代码)**:用一个独立单测直接构造
   `cldnn::qsa` primitive + 手工合成的恒等 `block_indices` 常量,验证退化分页在
   真实 GQA/compress_ratio 配置下和现有分页测试用例数值一致(复用
   `qsa_gpu_test.cpp` 的参考实现,只换输入构造方式)。
2. **阶段 1**:新增 `cldnn::qsa` 的 `skip_kv_write` 字段 + `qsa.cpp` 里 Q0
   `add_stage()` 的条件化包装,验证 `QSA`(分页模式)现有 47+ 个 gtest 全部不受影响。
3. **阶段 2a(已完成,见文末记录)**:新增 `ov::intel_gpu::op::QSASDPA`(`InputIdx`
   含 `PAST_LENS`/`SUBSEQUENCE_BEGINS`,支持 `MIXED`;`Config` 直接复用
   `ov::op::internal::QSA::Config`),core dev_api + gpu_opset 两层,参照 `QSA`/
   `GatedResidual` 已验证过的继承模式。**不含** ProgramBuilder lowering,尚不能
   参与真实图编译/GPU 执行。
   **阶段 2b(已完成,见文末记录)**:新增 `ov::intel_gpu::op::QSAIndexerCache`
   (`VariableExtension`,gpu_opset-only,无 core dev_api 层——和 `KVCache` 一样,
   因为它只会由未来的图融合 pass 内部产生,不是模型作者直接构造的算子)。
   `VariableExtension` 接口只支持单一 `Variable`,而这个 op 需要同时持有 raw-K
   和 summary 两份状态,因此只有 raw-K 走 `get_variable()`/`get_variable_id()`
   (接口契约),summary 通过新增的 `get_summary_variable()`/
   `get_summary_variable_id()` 显式暴露。**尚不含** beam 重排(4.9 节的
   `m_indirect`/`beam_table` 字段)、`trim`、动态 B 支持,也**不含**
   ProgramBuilder lowering——这些留给后续阶段。
4. **阶段 3a(已完成,见文末记录)**:新增 `ops/qsa_sdpa.cpp::CreateQSASDPAOp`
   (4.4 节的 lowering 逻辑,`REGISTER_FACTORY_IMPL(internal, QSASDPA)`),首次
   实现并验证了 `QSASDPA` 在真实 `ov::Model → ov::Core::compile_model("GPU") →
   infer()` 全流程下的端到端数值正确性(prefill/decode-continuation/MIXED 三种
   场景)以及动态 `B`/动态 `pa_block_size` 的显式拒绝。
   **阶段 3b(已完成,见文末记录)**:新增 `ops/qsa_indexer_cache.cpp::
   CreateQSAIndexerCacheOp`(`ReadValue → cldnn::qsa(prepare_only=true) →
   Assign`,`REGISTER_FACTORY_IMPL(internal, QSAIndexerCache)`,用
   `ov::intel_gpu::op::KVCache` 同款的"无 core dev_api、用别名让宏解析到
   gpu_opset 类型"手法,因为它和 `KVCache` 一样没有真实 `ov::op::internal`
   对应类),并用一个跨两次独立 `ov::InferRequest::infer()` 调用的端到端测试
   验证了 indexer raw-K/summary 状态确实通过 `Variable` 被正确持久化——这正是
   阶段 3a 交付时明确标注"未覆盖"的那个缺口,细节和踩过的坑见文末记录。
   `QSASDPAFusion` 图匹配 pass(识别分解形式的原始子图、重写成 `QSASDPA`/
   `QSAIndexerCache` 这一对算子,类比已有的 `QSAFusion` 之于分页版 `QSA`)仍
   **未实现**。
5. **阶段 4(按用户要求,本轮明确排除)**:新增 `QSABeamReorderOpt` pass(4.9 节的
   "及早物化" `Gather`),支持 beam search 场景;只依赖阶段 2 的
   `m_indirect`/`beam_table` 字段,可与阶段 3 并行开发。
6. **阶段 5(除 beam 相关外已完成,见文末记录)**:`qsa_gpu_test.cpp` 端到端测试
   (见第 6 节),含数值一致性交叉验证、`MIXED`(批内序列 `total_len` 相同的场景)、
   "跨调用改变 `B`"的显式拒绝行为验证(见第 7 节的限制)、`QSAIndexerCache` 的
   `Variable` 状态跨调用持久化验证。**不含** beam search 端到端测试(按用户要求
   排除)、批内序列 `total_len` 不同时的共享 S 轴 padding 场景(仍是一个独立的
   未实现场景,`QSAIndexerCache` lowering 完成后依然没有覆盖它)。
7. **阶段 6(可选)**:第 4.8 节的效率缓解手段,按实际 batch 场景需要决定是否做。

## 6. 测试计划

- **数值一致性**:构造同一组 (past_len, q_len, 权重) 输入,分别喂给 `QSA`(分页
  模式,恒等 block_indices,人工模拟同样的退化分页)和 `QSASDPA`(真实走 `KVCache`
  +新 op),断言输出逐元素一致(容差与现有 dpas vs scalar 一致,2e-2 绝对误差)。
- **状态生命周期**:验证 `reset_state()` 清空 `QSASDPA` 用到的全部 3 个
  `Variable`(主 K、主 V、raw-K、summary 共 4 个,若 K/V 各自独立 `Variable`)后,
  下一次调用等价于全新序列(`past_len=0`)。
- **Beam search**:验证 `QSABeamReorderOpt`(4.9 节)在检测到 `beam_idx` 重排后,
  四份状态(主 K、主 V、raw-K、summary)的显式 `Gather` 都被正确插入且重排一致;
  构造"连续两步 beam 重排、其中一步真实调换了 beam 顺序"的场景,断言输出与"预先
  手动按同一 beam 顺序重排全部四份状态、再走非 beam 路径"的参考结果一致。
- **MIXED**:构造同一个 batch 内一部分序列 `q_len=1`(decode)、一部分
  `q_len>1`(prefill/续写)的场景,验证 `PAST_LENS`/`SUBSEQUENCE_BEGINS` 正确复用
  `QSAStage::MIXED` 路径,输出与逐序列单独调用 `QSASDPA` 的结果一致。
- **动态 `B`(应被拒绝)**:验证同一个编译好的模型在相邻两次调用之间改变 batch 内
  序列数时,要么在 shape 校验阶段就报出清晰错误,要么行为在文档里被显式声明为
  未定义——不能静默产生错误数值(见第 7 节的限制)。
- **隔离回归**:`QSA`(分页模式)现有全部 gtest 在阶段 1-4 每一步之后原样跑一遍,
  确认零回归。
- **边界场景**:`S=0`(空历史)、`S` 恰好是 `R` 的整数倍(不产生因果尾部)、单 token
  decode(`QUERY` 只有 1 行)、batch 内序列数 `B` 固定但各序列历史长度差异很大
  (验证 dense `S` 轴 padding 处理正确,见 4.1 节)。

## 7. 已知限制(v1 明确不支持)

**动态 `B` 增长/缩减(运行时任意增删序列)不在 v1 支持范围内。** batch 内的序列数
`B` 在一次 `compile_model` 产出的可执行网络生命周期内被视为固定;需要变更 `B`
(例如更接近 vLLM 式 continuous batching 那种随时增删请求)必须重新构图/编译,或者
干脆改用现有的 `QSA`(分页模式)——那条路径本来就是为"调用方自行管理任意数量、任意
长度序列的物理页池"设计的,不受这个限制。

原因:退化分页把 `pa_block_size` 恒等地取成当前 batch 共享的稠密 `S` 轴长度、
`block_indices` 恒等映射到"每序列一页",这两个前提都隐含"`B`(序列数)和页数
一一对应、且这个对应关系在一次编译生命周期内不变"。支持任意增删序列意味着要么
(a) 按 batch 内最大 `S` 统一 `pa_block_size`(简单,但短序列浪费更多显存,且 `B`
本身仍需要能在图里变化,触及更深层次的动态 shape 问题),要么 (b) 彻底放弃
"退化"这个简化前提、走变长 page 的完整分页(等价于重新做一遍 `QSA` 已经做过的
事)。两者都超出本次"优先复用、快速交付"的设计目标,因此明确排除在 v1 之外,后续
如有需要再单独立项评估。

`QSASDPA` 的校验层应显式检查并拒绝"相邻两次调用的 batch 序列数改变"这种输入
(而不是让它静默产生错误结果)——具体检查点留到实现阶段确定(可以是 `KVCache`
`Variable` 的 batch 维度校验,也可以是 `QSASDPA` 自己的 `validate_and_infer_types`)。

## 8. 仍然待定的问题

1. **`max_seq_len` 上限与显存预算**:第 4.8 节的 `trim`/`max_seq_len_hint` 是否
   在首个交付版本就需要,取决于目标部署场景的 batch/上下文长度量级,建议先用
   第 2.2 节的公式对目标场景实际核算一遍再决定优先级。

~~2. Indexer raw-K/summary 的真正状态管理方案~~ —— **已于 2026-09-16 定案**,采用
方案二(`QSAIndexerCache` 原样复用 Q1 kernel,不外包给通用 `KVCache`),见 4.3 节
及文末"实现与验证记录"。方案二引入的"`QSASDPA` 侧 Q1 重复计算一次"这一确定性代价
已记录在 4.3 节末尾,不再是待定问题,而是已接受的设计权衡。

## 实现与验证记录

**2026-09-16,阶段 0 + 阶段 1,远程 BMG(Arc B580,`10.239.140.245`)验证:**

- 新增 `cldnn::qsa::skip_kv_write`(默认 `false`,纯可选构造参数,详见
  [qsa.hpp](../openvino.mx/src/plugins/intel_gpu/include/intel_gpu/primitives/qsa.hpp))
  + `qsa.cpp` 里 Q0(`qsa_kv_cache_update`)stage 的条件化 `add_stage`/`execute_stage`
  包装(详见 [qsa.cpp](../openvino.mx/src/plugins/intel_gpu/src/graph/impls/cm/qsa/qsa.cpp))。
  未新增/未修改任何 `.cm` kernel,未触及 `ov::op::internal::QSA`/`ov::intel_gpu::op::QSA`
  的 `Config`/`InputIdx`。
- 新增 5 个 GPU 单测(
  [qsa_gpu_test.cpp](../openvino.mx/src/plugins/intel_gpu/tests/unit/test_cases/qsa_gpu_test.cpp)):
  `degenerate_paging_prefill_matches_reference`、
  `degenerate_paging_prefill_continuation_matches_reference`、
  `degenerate_paging_decode_matches_reference`(验证 2.2 节的退化分页假设——
  `pa_block_size` 设成覆盖整段历史的单页、`block_indices` 恒等映射——在 prefill/
  chunked-prefill-continuation/decode 三种场景下都和 FP64 参考数值一致)、
  `skip_kv_write_matches_normal_path_on_degenerate_paging`、
  `skip_kv_write_matches_normal_path_on_decode`(验证 `skip_kv_write=true` +
  预先写好的主 K/V cache,和 `skip_kv_write=false` 走 Q0 正常写入,产出**逐字节
  相同**的输出,tolerance=0)。
- `run_qsa()`/`expect_qsa_matches()` 测试辅助函数各新增一个默认为 `false` 的
  `skip_kv_write` 参数(向后兼容,所有既有调用点不变)。
- 验证结果:新增 5 个测试全部 PASS;`--gtest_filter='*qsa*:*gated_residual*:
  -DISABLED_*'` 完整回归 60/60 PASS,零回归。
- **未实现**(当时超出交付范围):`ov::op::internal::QSASDPA`/
  `ov::intel_gpu::op::QSASDPA` 本体、`ops/qsa_sdpa.cpp::CreateQSASDPAOp` lowering、
  `QSASDPAFusion`/`QSABeamReorderOpt` 图变换 pass、indexer raw-K/summary 的状态
  管理算子。原因:发现的 4.2 节前提错误(indexer raw-K 不能走通用 `KVCache`)
  说明这些后续阶段的真实工作量比原设计估计的更大,需要先重新定案再继续实现。

**2026-09-16(同日,方案分析与选型后),`QSAIndexerCache` primitive 基础
(`prepare_only`),远程 BMG 验证:**

- 对比分析了两个候选方案(方案一:拆开 indexer K 投影与 raw-K 追加,让 raw-K 也走
  通用 `KVCache`;方案二:保持 Q1 融合不变,给 raw-K/summary 状态包一个专属
  `VariableExtension` 算子),**采纳方案二**,理由见 4.3 节(零新增 kernel、复用度
  最高、风险最低、与已验证的退化分页心智模型一致)。
- 新增 `cldnn::qsa::prepare_only`(默认 `false`,纯可选构造参数,与 `skip_kv_write`
  同一套字段处理:hash/equality/save/load 都已覆盖,详见
  [qsa.hpp](../openvino.mx/src/plugins/intel_gpu/include/intel_gpu/primitives/qsa.hpp))。
  为真时 `qsa.cpp` 只 `add_stage` Q1(`prepare`/`project_dpas`),Q0/Q2/Q3 全部不
  注册;`execute()` 在 Q1 完成后直接 `return`(详见
  [qsa.cpp](../openvino.mx/src/plugins/intel_gpu/src/graph/impls/cm/qsa/qsa.cpp))。
  同样未新增/未修改任何 `.cm` kernel。
- 新增 3 个 GPU 单测:`prepare_only_extends_raw_k_and_summary_from_scratch`、
  `prepare_only_extends_raw_k_and_summary_from_partial_history`、
  `prepare_only_matches_reference_with_degenerate_paging`——验证 `prepare_only=true`
  时,Q1 单独运行(无 Q0/Q2/Q3)在从零、从任意 partial 历史、以及退化分页三种场景
  下,都能把 raw-K/summary cache 正确扩展到和 FP64 参考完全一致的内容(逐 token/
  逐 block 比对,5e-2 容差,和其余 QSA 数值测试同一标准)。
- 验证结果:新增 3 个测试全部 PASS;`--gtest_filter='*qsa*:*gated_residual*:
  -DISABLED_*'` 完整回归 63/63 PASS(60 + 新增 3 个),零回归。
- **已知权衡,已记录**(见 4.3 节末尾):`QSASDPA` 自己的 `cldnn::qsa` 实例仍需要
  原样跑一遍 Q1 来算当次 indexer Q,Q1 因此在每次调用里跑两遍(一遍维护持久状态,
  一遍算当次值,幂等但重复计算)。这是方案二的确定性代价,不是本次发现的新问题。
- **仍未实现**:`QSAIndexerCache` 的 OV 算子本体(`VariableExtension`/`Variable`
  生命周期管理、beam_table 输出)、`ov::intel_gpu::op::QSASDPA` 及其
  `ops/qsa_sdpa.cpp` lowering、`QSASDPAFusion`/`QSABeamReorderOpt` 图变换 pass——
  本次只实现并验证了这些后续算子将要包装的 **cldnn primitive 层基础能力**
  (`skip_kv_write` + `prepare_only`),OV 图层(op 类、状态生命周期、
  ProgramBuilder 翻译、图匹配 pass)仍是后续会话的工作。

**2026-09-16(同日,阶段 2a),`ov::intel_gpu::op::QSASDPA` op 本体,远程 BMG 验证:**

- 新增 4 个文件,严格镜像 `QSA`/`GatedResidual` 已验证过的 core dev_api + gpu_opset
  继承模式(参见 `QSA_gr_dev_api_exposure` 相关经验):
  [qsa_sdpa.hpp](../openvino.mx/src/core/dev_api/openvino/op/qsa_sdpa.hpp)/
  [qsa_sdpa.cpp](../openvino.mx/src/core/src/op/qsa_sdpa.cpp)(`ov::op::internal::
  QSASDPA`,`Config` 直接复用 `ov::op::internal::QSA::Config`——不重复定义,几何
  校验通过 `validate_config()` 转调 `QSA::validate_config()`)与
  [qsa_sdpa.hpp](../openvino.mx/src/plugins/intel_gpu/include/intel_gpu/op/qsa_sdpa.hpp)/
  [qsa_sdpa.cpp](../openvino.mx/src/plugins/intel_gpu/src/plugin/transformations/op/qsa_sdpa.cpp)
  (`ov::intel_gpu::op::QSASDPA`,仅新增 `clone_with_new_inputs()` 覆盖以保持
  gpu_opset RTTI,其余全部继承)。`InputIdx` 与本文 4.1 节的 13 输入定义完全一致。
- 这两个文件都是**全新文件**,依赖 `src/core/CMakeLists.txt`/intel_gpu 侧的
  `dev_api/**/*.hpp`、`src/**/*.cpp` glob——按仓库既有 gotcha,新增文件必须先跑一次
  `cmake .` 重新配置,单纯 `make` 不会捡到它们;本次直接先跑了 `cmake .` 再
  `make`(未重复验证"不重新配置会漏掉新文件"这条,沿用既有记录,不是本次新验证)。
- **无 ProgramBuilder lowering**:没有创建 `ops/qsa_sdpa.cpp::CreateQSASDPAOp`,
  没有 `REGISTER_FACTORY_IMPL`。这个 op 目前只能在 C++ 里直接构造/做 shape
  推断,不能出现在真实可编译的 `ov::Model` 图里,也不能在真实 GPU 上跑一次
  `compile_model`/`infer()`——这是刻意的范围收敛,因为 `ops/qsa_sdpa.cpp` 那一层
  需要先解决"`QSASDPA` 自己的 `KEY`/`VALUE`(当次新 token)在 lowering 时该填什么"
  这类尚未定案的细节(main K/V 已经被上游 `KVCache` 追加过,`cldnn::qsa` 的固定
  17 输入契约里 `KEY`/`VALUE` 两个槽位在 `skip_kv_write=true` 时是死值,但仍需要
  合法 shape 的占位张量),留给阶段 3 专门处理,不在本次仓促决定。
- 新增 4 个纯 host 侧单测(无需 GPU,和 `config_validation_rejects_illegal_
  attributes`/`max_selected_tokens_reserves_the_tail` 同一测试风格):
  `qsasdpa_infers_output_shape_from_query`(构造 13 个输入,验证输出 shape/dtype
  推断正确)、`qsasdpa_rejects_wrong_input_count`(去掉一个输入,验证抛出
  `ov::Exception`)、`qsasdpa_clone_keeps_gpu_opset_type`(验证 `clone_with_new_
  inputs()` 不会把 gpu_opset 类型降级成 base internal 类型——`QSA`/`GatedResidual`
  暴露 dev_api 时踩过的同一个坑)、`qsasdpa_reuses_qsa_config_validation`(验证
  `QSASDPA::validate_config` 和 `QSA::validate_config` 对同一份非法配置给出相同
  拒绝结果)。
- 验证结果:新增 4 个测试全部 PASS;`--gtest_filter='*qsa*:*gated_residual*:
  -DISABLED_*'` 完整回归 67/67 PASS(63 + 新增 4 个),零回归。`ov_gpu_unit_tests`
  本身就会编译 `plugin/transformations/op/qsa_sdpa.cpp`(和 `qsa.cpp`/
  `gated_residual.cpp` 同目录,构建日志可见);另外按仓库已知的独立 gotcha(
  `ov_gpu_unit_tests` **不会**重新编译 `plugin/ops/*.cpp`,那里才是 factory
  注册冲突真正会暴露的地方)单独强制重建了 `openvino_intel_gpu_plugin` 真实插件
  目标,确认干净通过——本次未触碰 `REGISTER_FACTORY_IMPL`/`ops/*.cpp`,所以这个坑
  本来就不适用,但仍按标准流程验证了一遍。
- **仍未实现**(下一步):`ops/qsa_sdpa.cpp::CreateQSASDPAOp` lowering(含
  `KEY`/`VALUE` 占位张量的具体方案、`block_indices`/`block_indices_begins` 的
  常量合成、`REGISTER_FACTORY_IMPL` 注册)、`QSAIndexerCache` 的 OV 算子本体、
  `QSASDPAFusion`/`QSABeamReorderOpt` 图变换 pass、端到端 GPU 测试。

**2026-09-16(同日,阶段 2b),`ov::intel_gpu::op::QSAIndexerCache` op 本体,远程 BMG 验证:**

- 新增 2 个文件(仅 gpu_opset 层,**无 core dev_api 对应类**——和 `KVCache`/
  `ReadValue` 同理:这类"融合 pass 才会产生"的内部状态管理算子,从不由模型作者
  直接构造,不需要暴露到 dev_api):
  [qsa_indexer_cache.hpp](../openvino.mx/src/plugins/intel_gpu/include/intel_gpu/op/qsa_indexer_cache.hpp)/
  [qsa_indexer_cache.cpp](../openvino.mx/src/plugins/intel_gpu/src/plugin/transformations/op/qsa_indexer_cache.cpp)。
  `Config` 复用 `ov::op::internal::QSA::Config`(和 `QSASDPA` 一致);`InputIdx` 为
  `MIXED`/`INDEXER_QK_WEIGHT`/`INDEXER_Q_NORM_WEIGHT`/`INDEXER_K_NORM_WEIGHT`/
  `ROTARY_COS_SIN`/`PAST_LENS`/`SUBSEQUENCE_BEGINS`(7 个,严格对应 4.3 节注释里的
  输入列表,v1 不含 `beam_idx`)。
- **两个 Variable 的设计取舍**(实现前发现的、设计文档原先未明确的细节,本次定案):
  `ov::op::util::VariableExtension` 接口只支持单个 `Variable`(`get_variable()`/
  `get_variable_id()` 都是单数),但本 op 需要同时持有 raw-K 和 summary 两份持久
  状态。采用的做法:`m_variable`(继承自 `VariableExtension`)固定指向 raw-K 的
  `Variable`,summary 的 `Variable` 作为普通成员 `m_summary_variable` 存储,并新增
  `get_summary_variable()`/`get_summary_variable_id()`/`set_summary_variable()`
  显式访问器。下游(状态重置、beam 重排、未来的 ProgramBuilder lowering)需要同时
  处理两份状态时,必须用这组显式接口,不能只依赖通用的 `VariableExtension` 接口
  拿到全部状态——已在头文件注释里写清楚,避免日后被"看起来只有一个 Variable"误导。
- `validate_and_infer_types()` 参照 `ReadValue`(而不是 `KVCache`)的模式:输出
  shape/dtype 直接取自两个 `Variable::get_info()`,不依赖任何图输入的 shape——因为
  这个 op 和 `ReadValue` 一样是"纯状态驱动输出",不像 `KVCache` 那样有一个显式的
  "new_token_data" 输入去做 concat shape 推断(4.3 节已解释过:raw-K 的"新值"是
  Q1 kernel 内部算出来的,不是一个现成的图输入)。
- 新增 4 个纯 host 侧单测(同一风格,构造 `ov::op::util::Variable`/
  `VariableInfo` 走 `kv_cache_fusion_test.cpp` 里验证过的写法):
  `qsa_indexer_cache_infers_output_shapes_from_variables`、
  `qsa_indexer_cache_exposes_both_variable_ids`、
  `qsa_indexer_cache_rejects_wrong_input_count`、
  `qsa_indexer_cache_clone_preserves_both_variables`(验证 `clone_with_new_inputs()`
  正确保留两个 `Variable` 而不是只保留 `VariableExtension` 那一个)。
- 验证结果:新增 4 个测试全部 PASS;`--gtest_filter='*qsa*:*gated_residual*:
  -DISABLED_*'` 完整回归 **71/71 PASS**(67 + 新增 4 个),零回归;`cmake .`
  重新配置后 `make -j28 ov_gpu_unit_tests` 干净编译新文件;单独强制重建
  `openvino_intel_gpu_plugin` 确认干净通过(本次同样未触碰
  `REGISTER_FACTORY_IMPL`/`ops/*.cpp`,这个坑本来就不适用)。
- **仍未实现**(下一步):`QSAIndexerCache`/`QSASDPA` 的 ProgramBuilder lowering
  (`ops/qsa_indexer_cache.cpp`/`ops/qsa_sdpa.cpp`,含 `REGISTER_FACTORY_IMPL`
  注册)、`QSASDPAFusion`/`QSABeamReorderOpt` 图变换 pass、beam 重排(`m_indirect`/
  `beam_table`)、`trim`/动态 B 支持、端到端 `ov::Model` → `compile_model("GPU")` →
  `infer()` GPU 测试。这是当前设计里验证覆盖最大的缺口——只有走完 ProgramBuilder
  lowering 才能第一次在真实 GPU 上跑通完整的 `QSASDPA` 数据流。

**2026-09-16(同日,阶段 3a),`ops/qsa_sdpa.cpp::CreateQSASDPAOp` ProgramBuilder
lowering + 首次真实端到端 GPU 验证:**

- **关键澄清(实现过程中发现,已写入 4.4 节)**:`pa_block_size` 被
  `graph/impls/cm/qsa/qsa.cpp` 通过 `jit.make("QSA_PA_BLOCK_SIZE", ...)` 烘焙成
  CM kernel 的 **JIT 编译期常量**,不是运行时可变的量。这意味着 `QSASDPA` 的退化
  分页"页大小"(等价于批共享的最大序列长度容量)和批内序列数 `B` 都必须在
  `compile_model()` 时静态已知——这是"v1 不支持动态 B"这条已有限制的具体、可执行
  形式,把它变成了实现层面的硬约束而不只是文字声明。这一发现让 lowering 本身
  比原设计设想的简单得多(不需要任何 `ShapeOf`/`Range`/`Concat` 动态图手术,见
  4.4 节改写后的内容),但也意味着"批内序列历史长度不同、共享 S 轴按最长序列
  padding"这类场景仍需要额外工作(见下方"仍未实现")。
- 新增 [ops/qsa_sdpa.cpp](../openvino.mx/src/plugins/intel_gpu/src/plugin/ops/qsa_sdpa.cpp)
  (`CreateQSASDPAOp` + `REGISTER_FACTORY_IMPL(internal, QSASDPA)`,注册进
  [primitives_list.hpp](../openvino.mx/src/plugins/intel_gpu/include/intel_gpu/plugin/primitives_list.hpp));
  实现内容严格对应改写后的 4.4 节:静态 shape 校验(`KEY_CACHE_DENSE` 必须完全
  静态且其 S 轴等于 `config.pa_block_size`,`PAST_LENS` 批维度必须静态且与之
  一致,否则在图构建期 `OPENVINO_ASSERT` 拒绝)、`KEY_CACHE_DENSE`/
  `VALUE_CACHE_DENSE` 原样透传(形状本就与 `QSA::InputIdx::KEY_CACHE`/
  `VALUE_CACHE` 逐维度相同)、`INDEXER_RAW_K_CACHE_DENSE`/
  `INDEXER_SUMMARY_CACHE_DENSE` 各用一个静态 `cldnn::reshape` 插入 size-1 的
  head 轴、`block_indices`/`block_indices_begins` 用 host 端 C++ 循环直接算好的
  `cldnn::data` 常量(而不是运行时 `Range` 算子)、`QSA::InputIdx::KEY`/`VALUE`
  用极小的全零 `cldnn::data` 常量占位(`skip_kv_write=true` 时 Q0 从不读取它们,
  只需要满足 C++ 层面"存在一个合法输入"的构造要求)。
- **`qsa_inst`/CM 实现的配套修改**(`prepare_only` 场景下的输出别名,`阶段 2b`
  遗留的一处发现):`graph/qsa.cpp::calc_output_layouts` 新增
  `desc->prepare_only` 分支,返回 `[INDEXER_RAW_K_CACHE 输入的 layout,
  INDEXER_SUMMARY_CACHE 输入的 layout]` 而不是"attention 输出"那一个;
  `impls/cm/qsa/qsa.cpp::execute()` 在 `!use_attention()` 分支里新增
  `instance.set_output_memory(instance.dep_memory_ptr(...), false, 0/1)`,把
  这两个声明的输出**别名**到 Q1 已经原地修改过的同一块内存,而不是另外拷贝一份
  ——让下游(未来 `QSAIndexerCache` 的 `Assign`)获得一条真实的图依赖边可以调度,
  这是让 `QSAIndexerCache` 未来能够正确参与 cldnn 依赖图调度所必需的基础设施,
  在验证 `QSASDPA` lowering 的过程中提前做了(不依赖 `QSAIndexerCache` 本体)。
  `cldnn::qsa` 构造函数相应改为 `prepare_only=true` 时自动把 `num_outputs` 设为
  2(此前是恒为 1),现有 3 个 `prepare_only` 单测因此需要重新验证——已确认仍然
  PASS。
- **两个与"通过真实 `ov::Model` 编译"相关、之前从未被本项目触发过的既有 GPU
  plugin 行为**(均已定位、正确处理,不是新增的 bug):
  1. `ov::Core::compile_model(...)` 走的是**动态加载的 `libopenvino_intel_gpu_
     plugin.so`**,和 `ov_gpu_unit_tests` 静态链接的 `openvino_intel_gpu_graph`
     库是**两个不同的产物**——本会话至此为止的每一次验证都只用到了原始
     cldnn::topology API(不经过 `ov::Core`),因此只需要重建
     `ov_gpu_unit_tests`。这是本会话第一次写"真正调用 `ov::Core::compile_model`
     并读、写 GPU 数据"的测试,踩到了"改了 `graph/qsa.cpp` 之后只 `make
     ov_gpu_unit_tests`,报错信息却像是旧代码在运行"——根源是 `libopenvino_
     intel_gpu_plugin.so` 没有被同一次 `make` 命令捎带重建。**新记忆点**:任何
     经由 `ov::Core`/`compile_model` 的测试,必须显式 `make -j28
     ov_gpu_unit_tests openvino_intel_gpu_plugin`(两个目标都要),不能只建
     `ov_gpu_unit_tests`。
  2. `ProgramBuilder::use_new_shape_infer()`(内部属性
     `GPU_ALLOW_NEW_SHAPE_INFER`,默认 `false`,只在图里存在动态 shape 时才会被
     置真)为 `false` 时,`plugin/ops/parameter.cpp::CreateParameterOp` 会把
     RANK < 4 的 `ov::op::v0::Parameter` 静态 shape 一律**补齐到 4 维**(在末尾
     追加 `1`)。由于本次测试的每个输入都是完全静态 shape,新模型自然落进
     这个"legacy 补齐"分支,导致 `graph/qsa.cpp`/`impls/cm/qsa/qsa.cpp` 里两处
     "rank 必须恰好等于 2"的断言(`calc_output_layouts`、`token_pitch()`)报错。
     该内部属性本身**不能**通过公开的 `ov::Core::compile_model(model, "GPU",
     properties)` 设置(`Couldn't set unknown property`,只有测试专用的
     `ExecutionConfig`/`get_test_default_config` 内部路径能设),因此修的是
     断言本身而不是绕开这个属性:两处断言都改为接受"rank ≥ 2 且第 2 维之后
     全部恰好等于 1",值语义不变(index 0/1 始终是正确的 tokens/stride),只是
     不再要求"恰好是 rank 2 这一种具体的表示形式"。这个改动对现有 rank-2 场景
     100% 向后兼容(zero regression 已验证)。
- 新增 4 个使用真实 `ov::Model` → `ov::Core::compile_model("GPU")` →
  `ov::InferRequest::infer()` 的端到端测试(全部走 `CreateQSASDPAOp` 这条新
  lowering 路径,是本设计文档第一批真正意义上的端到端 GPU 测试):
  `qsasdpa_ov_model_prefill_matches_reference`(past_len=0,q_len=64)、
  `qsasdpa_ov_model_decode_continuation_matches_reference`(past_len=63,
  q_len=1)、`qsasdpa_ov_model_mixed_batch_matches_reference`(2 个共享权重的
  序列,一个从零 prefill、一个续写 1 步 decode,同批调用,验证 `QSAStage::MIXED`
  在 `QSASDPA` 路径下与 `QSA`(分页模式)复用同一套 host 端派发代码这一论断)、
  `qsasdpa_rejects_dynamic_key_cache_shape`(`KEY_CACHE_DENSE`/`PAST_LENS` 声明
  为动态 shape,断言 `compile_model` 在图构建期以 `ov::Exception` 明确拒绝,而
  不是编译通过后产生错误数值)。前三个测试里,主 K/V 与 indexer raw-K/summary
  缓存全部由普通 dense `ov::op::v0::Parameter` 直接提供(不经过
  `ov::intel_gpu::op::KVCache`/`QSAIndexerCache`)——这是设计里明确允许、合法的
  "调用方自行管理全部缓存状态"用法,刻意选择它是为了把"`QSASDPA` 这个新
  lowering 本身是否正确"和"`KVCache`/`QSAIndexerCache` 集成是否正确"这两个
  问题解耦,分别验证。
- 验证结果:新增 4 个端到端测试全部 PASS;`--gtest_filter='*qsa*:*gated_residual*:
  -DISABLED_*'` 完整回归 **75/75 PASS**(71 + 新增 4 个),零回归;
  `openvino_intel_gpu_plugin` 与 `ov_gpu_unit_tests` 联合重建确认干净通过。
- **仍未实现**(下一步,按用户要求本轮明确排除 beam 相关部分):
  1. `QSAIndexerCache` 自己的 `ops/qsa_indexer_cache.cpp` ProgramBuilder
     lowering(含 `Variable` 状态持久化)——实现过程中做了充分的技术预研但
     决定不在本轮仓促实现,原因见下方"技术预研记录"。
  2. `QSASDPAFusion` 图匹配 pass(识别分解形式的原始子图、重写成 `QSASDPA`/
     `QSAIndexerCache` 这一对算子,类比已有的 `QSAFusion` 之于分页版
     `QSA`)——`QSAFusion` 本身的匹配逻辑相当可观(参见
     `qsa_gr_fusion.cpp`),一个 SDPA 风格的对应版本是独立、量级相近的工作,
     本轮未启动。
  3. 批内序列 `total_len` 不同时的共享 S 轴 padding 场景(MIXED 端到端测试
     目前要求批内每个序列 `total_len` 相同,规避了这个问题)。
  4. `QSABeamReorderOpt` pass 与 beam search 端到端测试(按用户要求本轮明确
     排除,不在范围内)。
  5. `trim`/动态淘汰(第 4.8 节的可选缓解手段,量级较小,未评估优先级)。

  **技术预研记录(`QSAIndexerCache` lowering 为什么没有仓促实现)**:分析过程中
  发现,若延续"退化分页 `pa_block_size` 每次调用等于当前总长度"的原始设想,
  `QSAIndexerCache` 的 `Variable` 状态需要在每次调用之间**增长**(不同于本节
  确认的"`pa_block_size` 是编译期固定容量"这一前提)——若坚持"零浪费,精确
  等于当前长度"的增长语义,需要在图里插入
  `ReadValue → Concat(old, zero_pad) → [qsa(prepare_only) 原地写入] → Assign`
  这样的动态图手术(且 zero-pad 的长度依赖批内每个序列的历史长度,MIXED 场景
  下还不是一个简单的标量)。而按本节澄清的"`pa_block_size` 编译期固定"这一
  前提,`QSAIndexerCache` 的实现其实可以大幅简化:`Variable` 直接按固定容量
  `[B, pa_block_size, Di]` 声明,`ReadValue`(0 输入,首次访问按声明 shape
  自动零填充)的原始输出**不需要任何 Concat/padding 即可直接**作为
  `INDEXER_RAW_K_CACHE` 送进 `qsa(prepare_only=true)`(Q1 按 `past_lens`
  写入正确偏移,其余仍是零),`Assign` 直接写回同一个固定形状——不需要新的
  primitive_inst 级别基础设施,`cldnn::read_value`/`cldnn::assign` 这两个
  已有的通用 primitive(`impls/cpu/read_value.cpp`/`assign.cpp`,已被
  `tests/unit/test_cases/variable.cpp` 验证过在多次 `network.execute()` 之间
  正确持久化状态)即可直接复用,不需要模仿 `kv_cache_inst` 那样派生新的
  `typed_primitive_inst<qsa>` + `memory_state::variable` 组合(那是`KVCache`
  为了支持"精确增长、不过量分配"这一更难的语义才需要的重量级机制,
  `QSAIndexerCache` 采用"固定容量预留"后不再需要)。这条路径本身在分析上是
  可行的、风险可控的,只是本轮时间没有覆盖到实现+验证,留作下一步的第一
  候选任务(比 `QSASDPAFusion` 更小、更独立、不依赖任何未决问题)。

**2026-09-16(同日,阶段 3b),`ops/qsa_indexer_cache.cpp::CreateQSAIndexerCacheOp`
ProgramBuilder lowering + 跨两次 infer() 调用的真实状态持久化验证:**

用户明确指出:阶段 3a 的三个 QSASDPA 端到端测试(prefill/decode-continuation/
MIXED)都只是把 indexer raw-K/summary 缓存的"历史部分"用 FP64 参考实现在 host 端
直接算出来填进普通 `Parameter`,不经过任何真实 GPU 调用产生,因此完全没有验证
"indexer cache 本身是如何被维护出来的"——这正是上面"技术预研记录"里分析过、但
尚未实现验证的那部分。本次按分析的方案实现并验证。

- **实现前的一处设计修正**:`QSAIndexerCache::InputIdx` 原来 7 个输入遗漏了
  `POSITION_IDS`。Q1 的 RoPE 需要绝对位置,一般情况下不能从 `past_lens` 反推
  (mRoPE 场景尤其不行),不像 `QUERY`/`KEY`/`VALUE` 那样是真正的死值——补进
  `InputIdx`(现为 8 个),`ov::intel_gpu::op::QSAIndexerCache` 头文件/host 侧
  测试同步更新。
- 新增 [ops/qsa_indexer_cache.cpp](../openvino.mx/src/plugins/intel_gpu/src/plugin/ops/qsa_indexer_cache.cpp)
  (`CreateQSAIndexerCacheOp` + `REGISTER_FACTORY_IMPL(internal, QSAIndexerCache)`,
  注册进 primitives_list.hpp)。`QSAIndexerCache` 和 `KVCache` 一样没有真实
  `ov::op::internal` 对应类,所以用 `KVCache` 同款的 `using QSAIndexerCache =
  ov::intel_gpu::op::QSAIndexerCache;` 别名让宏解析——这条路径本身安全,不会
  重演 QSA/GatedResidual 那次"别名 vs 真实 dev_api 类冲突"的坑,因为这里根本没有
  真实 dev_api 类与别名竞争。
- **lowering 内容**:`ReadValue`(0 输入,raw-K/summary 各一个,首次访问零填充,
  声明形状固定为 `[B, pa_block_size, Di]`/`[B, pa_block_size/r, Di]`,和"技术
  预研记录"里分析的完全一致)→ reshape 成 4D 喂给 `cldnn::qsa(prepare_only=true)`
  → `Assign`(2 个,各自写回对应 Variable)。`QUERY`/`KEY`/`VALUE`/`KEY_CACHE`/
  `VALUE_CACHE` 全部合成(iota/零常量),`MIXED`/`INDEXER_QK_WEIGHT`/
  `INDEXER_Q_NORM_WEIGHT`/`INDEXER_K_NORM_WEIGHT`/`ROTARY_COS_SIN`/`PAST_LENS`/
  `SUBSEQUENCE_BEGINS`/`POSITION_IDS` 原样透传。
- **多输出算子的 canonical primitive 映射**:`QSAIndexerCache` 有 2 个 OV 声明
  输出,但 lowering 内部还要额外插入 2 个 `Assign`(它们添加在 qsa primitive
  *之后*,而 `ProgramBuilder::add_primitive(op, prim)` 每次都会把
  `primitive_ids[op 的 canonical id]` 重新指向刚添加的那个 prim)。解决方式:
  在函数末尾显式执行 `p.primitive_ids[layer_name] = layer_name;`,把 canonical
  映射改回指向 `cldnn::qsa` 本身(它的 id 恰好等于 `layer_name`,且它的两个真实
  输出端口 0/1 就是 raw_k_dense/summary_dense),这样任何下游消费者(未来的
  `QSASDPAFusion`)通过 `GetInputInfo()` 解析到的仍然是正确的两个端口。
- **实现过程中依次发现并修复的 3 个 bug(均已验证,未回归)**:
  1. **"prepare map buffer is too small"**:`get_internal_buffer_descs()`(Q1
     自己的工作缓冲区大小估计)用 `QUERY` 输入的行数(不是 `MIXED` 的)当作
     "本次调用的 token 数"——原以为 `QUERY` 在 prepare_only 模式下彻底是死值,
     给它一个固定 `[1, ...]` 的哑元常量,但它的**行数**(不是内容)实际会被
     读取用于缓冲区定容量,而这个行数在不同调用之间是动态变化的(跟 `MIXED`
     一样)。修复:直接把 `MIXED` 自己的图输出复用为 `QUERY` 槽位的实参(行数
     天然一致,列宽不匹配无所谓,因为没有任何代码读 `QUERY` 的列宽或内容)。
  2. **Assign 与 Variable 的 layout 不匹配**(`variable layout: 1x16x16` vs
     `assign output layout: 1x1x16x16`):`cldnn::qsa` 在 `prepare_only` 模式
     下的两个声明输出,原本直接照抄了 4D 分页态输入(`INDEXER_RAW_K_CACHE`/
     `INDEXER_SUMMARY_CACHE`)的 layout,而 `QSAIndexerCache` 的 `Variable`
     是按 3D 稠密态声明的——修复:改 `graph/qsa.cpp::calc_output_layouts` 的
     `prepare_only` 分支,把声明的输出 shape 从"直接复制 4D 输入 layout"改成
     "去掉恒为 1 的 head 轴,声明成 3D"。这个改动是安全的,因为 `execute()`
     的输出别名(`set_output_memory`)只是指针操作,声明的 rank/shape 本来
     就和底层字节完全无关,3D 还是 4D 只是"怎么描述同一块内存"的自由选择——
     选 3D 之后,`QSAIndexerCache` 的 lowering 就不再需要额外的"4D 转回 3D"
     reshape,直接用 qsa 的输出给 `Assign` 用。
  3. 上述第 2 点修复前的一次中间尝试(在 `Assign` 前插另外两个独立的 3D
     reshape 节点)会破坏"下游消费者需要用一个 primitive 的两个端口拿到两路
     输出"这个假设(两个独立 reshape 天然是两个独立、各自单端口的 primitive,
     没有办法用同一个 `primitive_ids[canonical_id]` 映射表示"端口 0 指向
     reshape A,端口 1 指向 reshape B")——中间状态被完全替换掉,不保留在
     最终代码里,记录在这里是为了说明为什么第 2 点的修复方式(改
     `calc_output_layouts`,而不是加 reshape)才是对的。
- 新增 1 个真正跨两次独立 `ov::InferRequest::infer()` 调用的端到端测试
  `qsa_indexer_cache_ov_model_persists_state_across_two_infer_calls`:同一个
  编译好的模型、同一个 `infer_request`,先用 `past_len=0, q_len=20` 调用一次
  (只喂这 20 个 token),再用 `past_len=20, q_len=20` 调用第二次(只喂另外
  20 个新 token,**绝不重复提交第一次的 token**)。第二次调用后读取的
  raw-K/summary 输出,同时对 40 个 token 的**全部**位置(含前 20 个,只可能来自
  第一次调用通过 `Variable` 持久化下来的状态)和参考实现比对,5e-2 容差,逐 token
  /逐 block 通过。这是第一个真正验证"indexer cache 是如何被维护"的测试,直接
  回应了用户对阶段 3a 端到端测试覆盖缺口的提问。
- 验证结果:新增 5 个测试(1 个多输入校验测试因 `InputIdx` 从 7 变 8 同步更新)
  全部 PASS;`--gtest_filter='*qsa*:*gated_residual*:-DISABLED_*'` 完整回归
  **76/76 PASS**(75 + 新增 1 个净增,原有 4 个 host 侧测试同步更新未新增计数),
  零回归;`cmake .` 重新配置后 `ov_gpu_unit_tests`/`openvino_intel_gpu_plugin`
  联合构建确认干净通过。
- **仍未实现**:`QSASDPAFusion` 图匹配 pass(本轮唯一剩下的、非 beam 相关的
  阶段 3 工作项)、批内序列 `total_len` 不同时的共享 S 轴 padding(`QSAIndexerCache`
  和 `QSASDPA` 的 MIXED 测试目前都要求批内 `total_len` 一致,这个限制在两者的
  lowering 都完成后依然存在,是一个独立的、尚未评估优先级的场景)、beam 重排
  相关的一切(按用户要求本轮不做)。

**2026-09-16(同日,阶段 3c),`QSASDPAFusion` 图匹配 pass,基于真实模型
`models/ov-Qwen3.8-Flash-Next-0.2B-A0.2B/qwen4_exp_text.xml`(4 层,第 3 层为
QSA `self_attn`+`indexer`)的实际算子拓扑落地,远程 BMG 构建验证(纯 OV 图变换,
不涉及 cldnn/CM,故本轮验证仅需 `ov_gpu_unit_tests`/`openvino_intel_gpu_plugin`
编译+跑 host 侧单测,未使用真实 GPU 执行):**

- **对真实导出图的关键发现**(决定了本 pass 的能力边界,而非猜测):
  - indexer 分数信号(`ReLU→ReduceSum→Divide(1/√Di)→Select(因果掩码)→TopK`)与
    `QSAFusion` 已有 pattern 逐节点吻合(`Relu_1436→ReduceSum_1437→Divide_1439→
    Select_1452→Minimum_1454(clamp)→TopK_1455`),证实该 pattern 具备真实代表性。
  - 主 K/V 的 `ReadValue`/`Assign` 变量(`past_key_values.3.key/value`)声明形状为
    `[?,2,?,256]`——**batch 和 seq 两个维度都是动态的**,不满足 `QSASDPA`/
    `cldnn::qsa` 的 `pa_block_size` 编译期常量约束(`ops/qsa_sdpa.cpp::
    CreateQSASDPAOp` 已有 `OPENVINO_ASSERT` 强制拒绝)。
  - indexer raw-K 状态(`...indexer.raw_keys_cache`,形状 `[?,?,128]`,稠密 3D,
    与 `QSAIndexerCache` 自身的 Variable 约定吻合)**存在**,但**不存在**独立的
    block-summary 状态变量——每次调用都从头对全量 raw-K 历史做 `ReduceMean` 池化
    重算 summary(`ReduceMean_1329` 等),这正是 `QSAIndexerCache` 的 Q1 增量
    kernel 所要消除的冗余计算,印证了本 pass 存在的意义。
  - 主注意力是唯一一处 `ov::op::v13::ScaledDotProductAttention`,由 TopK 选中的
    block 索引经 `Equal`/`GreaterEqual`/`LessEqual`/`LogicalAnd`/`LogicalOr`/
    `Select`+`Add` 构造出的**加性掩码**驱动(掩码式稀疏注意力,不是物理 KV block
    `Gather`)。
  - RoPE 通过 `full_cos`/`full_sin` 两个**独立、未交织的 `Parameter`**(而非单个
    已打包 `[positions, rotary_dim]` 常量表)+ `rotate_half` 分解公式实现——与
    `QSA`/`QSASDPA` 的 `ROTARY_COS_SIN`(`(cos,sin)` 交织对打包成一张表)输入
    约定不兼容,若要复用需要在融合时新插入图节点重新打包,风险与工作量都很大。
  - `beam_idx` Parameter 存在,且每条 `ReadValue→Concat` 路径中间都插了一次
    `Gather`(beam 重排),证实真实导出已经统一走"eager materialize"路径——但
    beam 相关内容按用户要求本轮不做,不影响以下设计。
  - 关键权重(`index_qk_proj.weight`、`q/k_layernorm.weight`、`q/k/v/o_proj.weight`
    等)均为独立具名 `Constant`,未被融合/混淆,形状均已用 `grep` 逐一核实
    (如 `index_qk_proj.weight=[640,256]` ⟹ `Di=128,Hidx=4`;`q_proj.weight=
    [4096,256]` ⟹ 含 output-gate 融合,`Hq=8`;`k/v_proj.weight=[512,256]` ⟹
    `Hkv=2,Dh=Dv=256`),证明"不靠字面权重名匹配、只靠结构化形状证明"的方式
    在这个真实模型上是可行的。
- **据此定的 v1 能力边界**(与 `QSAFusion` 一样,凡无法结构化证明的一律
  保守放弃、不猜测,不修改 `has_output_gate`/`q_proj`/`o_proj` 等 QSA 本不
  拥有的周边算子):
  1. 只在主 K/V 的 `ReadValue` 声明形状**完全静态**且 `batch==1` 时才替换
     ——这与 `qwen4_exp_text.xml` 的真实动态形状**正确地**不匹配,pass 在该文件
     上会正确地保留原图(这是预期行为,不是缺陷)。
  2. 只在 indexer RoPE 存在**单个**已打包 `[positions, rotary_dim]` 张量时才
     替换——`full_cos`/`full_sin` 两个独立张量的场景不满足,同样正确保留原图。
     图内重新打包 RoPE 表是明确列为未来增量,v1 不做。
  3. `q_proj`/`k_proj`/`v_proj`/`o_proj` 与任何 output-gate 逻辑完全不touched:
     `QSASDPA` 的 `QUERY`/`KEY_CACHE_DENSE`/`VALUE_CACHE_DENSE` 直接复用原
     `ScaledDotProductAttention` 节点自身的 Q/K/V 输入(该输入本就已经是
     投影+RoPE 之后、且(若有)gate 已被切走之后的张量),只替换 SDPA 节点
     本身的输出(reshape 回原 `[1,Hq,T,Dv]` 形状),下游 sigmoid-gate/`o_proj`
     完全无感知。
  4. `PAST_LENS`/`SUBSEQUENCE_BEGINS`/`POSITION_IDS` 通过 `ShapeOf`/`Gather`/
     `Range`/`Concat` 现场合成(`past_len=ShapeOf(raw_k_read_value)[1]`,
     `T=ShapeOf(mixed)[0]`,`position_ids=Range(past_len, past_len+T)`)——
     只支持 batch=1 单序列场景,与"static B" 的前提一致。
- **代码改动**:
  - 重构 [qsa_gr_fusion.cpp](../openvino.mx/src/plugins/intel_gpu/src/plugin/transformations/qsa_gr_fusion.cpp):
    把 indexer 分数信号的 pattern 构建(`build_qsa_indexer_pattern`)和验证
    (`verify_qsa_indexer_signature`)从 `QSAFusion` 构造函数中抽成共享的匿名
    命名空间函数,`QSAFusion` 本身改为调用这两个共享函数,行为完全不变(见下方
    回归结果)。新增 3 个有界 BFS 图遍历辅助(`find_ancestor<T>`/
    `find_descendant<T>`/`find_ancestor_if`),用于在不假设精确分解形状的前提下
    定位 `MatMul`/`ReadValue`/`ScaledDotProductAttention`/`ReduceMean`/`Constant`
    等"稀有、有区分度"的节点,再对**找到的节点自身**的形状/属性做结构化证明
    ——与 `QSAFusion` "结构证明、不猜测"的原则一致。
  - 新增 `QSASDPAFusion : public ov::pass::MatcherPass`(声明见
    [qsa_gr_fusion.hpp](../openvino.mx/src/plugins/intel_gpu/src/plugin/transformations/qsa_gr_fusion.hpp)),
    并加入 `QSAGatedResidualFusion` 容器。
- **新增测试**(
  [qsa_gr_fusion_test.cpp](../openvino.mx/src/plugins/intel_gpu/tests/unit/transformations/qsa_gr_fusion_test.cpp)):
  构造了一个 T=1(decode 单 token)、batch=1 的合成图,结构上镜像真实模型的关键
  要素(共享 `index_qk_proj` MatMul、indexer 独立 `ReadValue` 状态、block-mean
  池化、`QSAFusion` 已验证过的 indexer 分数信号、`ScaledDotProductAttention` +
  `ReadValue` 管理的主 K/V)。3 个新测试:
  - `qsa_sdpa_fusion_replaces_with_qsasdpa_and_indexer_cache`:一切均可证明且
    静态时,断言原 `ScaledDotProductAttention` 节点被移除、新增恰好 1 个
    `QSASDPA` + 1 个 `QSAIndexerCache`,且 `Config`(`heads_num`/`kv_heads_num`/
    `k_head_size`/`v_head_size`/`hidden_size`/`indexer_heads`/`indexer_head_dim`/
    `indexer_rotary_dim`/`compress_ratio`/`block_topk`/`budget`/`pa_block_size`)
    与合成图的真实几何完全一致,并断言 raw-K `Variable` 是**原样复用**(而非
    新建)、summary `Variable` 是新建且 id 不同。
  - `qsa_sdpa_fusion_bails_out_on_dynamic_kv_shape`:主 K/V `Variable` 声明为
    动态 S(直接镜像 `qwen4_exp_text.xml` 的真实情况),断言不替换、原 SDPA
    保留。
  - `qsa_sdpa_fusion_bails_out_without_packed_rotary_table`:不提供已打包
    RoPE 表(直接镜像 `full_cos`/`full_sin` 两张独立表的真实情况),断言不
    替换、原 SDPA 保留。
- **验证结果**:`ov_gpu_unit_tests`/`openvino_intel_gpu_plugin` 远程 BMG 联合
  构建干净通过;`--gtest_filter='qsa_gr_fusion.*'` 13/13 PASS(10 个既有
  `GatedResidualFusion`/`QSAFusion` 测试零回归 + 3 个新增全部 PASS);
  `--gtest_filter='*qsa*:*kv_cache*'` 更广回归 67 PASS / 0 FAILED / 47 SKIPPED
  (跳过的均为需要真实 GPU 执行的 `qsa_gpu_test`/`smoke_*` 用例,本轮改动是纯
  host 侧图变换,未涉及 cldnn/CM,不需要、也未使用真实 GPU 验证)。
- **仍未实现 / 明确列为 v1 之外**:图内重新打包 RoPE 表(支持
  `full_cos`/`full_sin` 分离表的场景)、动态 B/S 支持、批内多序列(ragged/
  continuous-batching)、beam 重排(按用户要求本轮不做)。这些是 `QSASDPAFusion`
  当前已知的能力边界,而不是本轮遗留的 bug——`qwen4_exp_text.xml` 按当前实际
  导出方式,就是会被本 pass 正确地跳过,不会被错误融合。

