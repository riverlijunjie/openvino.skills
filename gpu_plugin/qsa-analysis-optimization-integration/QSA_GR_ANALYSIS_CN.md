# Qwen3.8-Flash-Next：QSA 与 Gated Residual 实现分析

> 本文基于：
>
> - `/home/ov2022/workspace/qsa/Qwen3.8-Flash-Next/tech_report.pdf`
> - SGLang 分支 `qwen4-main-squashed` 中的 Qwen4-Exp 实现
>
> 当前 SGLang 分支中，Qwen3.8-Flash-Next 主要以 **`Qwen4Exp`** 命名实现。

## 1. 整体架构

Qwen3.8-Flash-Next 是一个混合架构模型：

- 每四层中包含三层 Gated DeltaNet（GDN）和一层全局 attention；
- 在 continued pretraining 阶段，全局 attention 层替换为 Qwen Sparse Attention（QSA）；
- residual stream 扩展为四个 branch；
- 每个 attention/MLP 子层都通过独立的 Gated Residual（GR）读取和写回；
- 额外使用一个 n-gram embedding 层增加模型容量。

三种机制的职责不同：

```text
GDN：以线性复杂度维护 recurrent memory
QSA：从长上下文中检索重要 token
GR ：控制多路 residual branch 的读写和跨层信息流
```

QSA 主要解决长上下文 attention 的计算和访存问题；GR 主要解决单一 residual stream 容量有限、早期层信息容易被后续层覆盖的问题。

## 2. SGLang 中的实现对应关系

| 模型结构 | SGLang 实现 |
|---|---|
| QSA indexer | `python/sglang/srt/layers/attention/qsa/qsa_indexer.py` |
| QSA 压缩、top-k、token 展开 | `python/sglang/srt/layers/attention/qsa/kernel.py` |
| QSA sparse GQA / FlashAttention 路径 | `python/sglang/srt/layers/attention/qsa/sparse_attn.py` |
| QSA metadata、page table、pending ring | `python/sglang/srt/layers/attention/qsa/metadata.py`、`qwen_sparse_attn_backend.py` |
| QSA KV cache | `python/sglang/srt/mem_cache/qsa_kv_pool.py` |
| Gated Residual 数学实现 | `python/sglang/srt/layers/hyperconnection.py` |
| Gated Residual fused mix kernel | `python/sglang/srt/layers/hc_mix_triton.py` |
| Qwen3.8 模型层接入 | `python/sglang/srt/models/qwen4_exp.py` |
| Qwen3.8 配置 | `python/sglang/srt/configs/qwen4_exp.py` |

典型配置包括：

- `hc_count = 4`：四个 residual branches；
- `hc_lowrank = 320`：GR read gate 的低秩 bottleneck；
- `indexer_budget = 2048`：每个 query 的 token budget；
- `indexer_compress_ratio = 4`：每四个 token 压缩成一个 micro-block；
- `indexer_budget / indexer_compress_ratio = 512`：每个 query 最多选择 512 个完整 blocks。

---

# 3. QSA：Qwen Sparse Attention

## 3.1 QSA 的基本思想

QSA 不是简单截断 KV cache，也不是直接把 compressed key 用作最终 attention。它分成两个阶段：

1. **Compressed Lightweight Indexer**：对压缩后的 key blocks 做重要性评分，选择 top-k blocks；
2. **Sparse Core Attention**：将选中的 blocks 展开成原始 token indices，只对这些 token 的真实 K/V 执行 attention。

整体路径为：

```text
hidden states
     │
     ├── indexer Q/K projection
     │       │
     │       ├── token key 按 micro-block 平均池化
     │       ├── query 与 compressed key 做 block scoring
     │       ├── top-k block selection
     │       └── block 展开为原始 token indices
     │
     └── 主 attention Q/K/V projection
             │
             └── 只读取被选中的 token K/V
```

如果序列长度为 $n$、compression ratio 为 $r$，indexer 处理的 key 数量从 $n$ 降到约 $n/r$，因此 indexer 复杂度从：

$$
O(n^2)
$$

降为：

$$
O\left(\frac{n^2}{r}\right)
$$

最终 core attention 只处理固定数量的 token，而不是完整历史。

## 3.2 Indexer 与主 attention 是两套投影

在 `QSAIndexer.project_qk()` 中：

```python
qk, _ = self.index_qk_proj(hidden_states)
```

这套投影不是主 attention 的 `q_proj/k_proj`，而是 indexer 的独立投影：

- 前半部分生成 indexer query；
- 后半部分生成 token key；
- indexer query 用于 block scoring；
- token key 用于生成 compressed key。

因此：

- indexer Q/K 只负责回答“哪些位置重要”；
- 主 attention Q/K/V 负责真正计算模型输出。

这使得 indexer 可以使用更少的 query heads 和更低的计算量，而不需要牺牲主 attention 的表达能力。

## 3.3 Compressed Lightweight Indexer

报告中 indexer 使用 MQA：

- $H$ 个 query heads；
- 一个共享 key head；
- Qwen3.8 最终使用 4 个 indexer query heads；
- indexer head dimension 为 128；
- partial RoPE 作用于其中 64 个维度。

对 token hidden state $x_i$，indexer 的 query/key 为：

$$
\tilde q_{i,h}=\operatorname{RMSNorm}(W^Q_hx_i)
$$

$$
 k_i=W^Kx_i
$$

与主 attention 相比，indexer 使用 MQA 和更少的 query heads，因此成本较低。

## 3.4 先压缩，再应用 RoPE

设 compression ratio 为 $r$，第 $b$ 个 block 从位置 $p_b=br$ 开始，覆盖：

$$
 p_b,p_b+1,\ldots,p_b+r-1
$$

QSA 先对未应用 RoPE 的 token key 做平均池化：

$$
\bar k_b=
\operatorname{RMSNorm}\left(
\operatorname{AvgPool}(k_{p_b:p_b+r-1})
\right)
$$

之后才应用 partial RoPE：

$$
q_{i,h}=\operatorname{PRoPE}(\tilde q_{i,h},i)
$$

$$
\bar k_b=\operatorname{PRoPE}(\bar k_b,p_b)
$$

其中：

- query 使用自己的 token position $i$；
- compressed key 使用 block 起始位置 $p_b$。

先压缩再加 RoPE 的原因是：如果先对带有不同旋转相位的 token key 做平均，平均结果会混合不同的 positional phase。先获得 content summary，再给整个 summary 分配单一 block position，可以避免该问题。

SGLang 中对应逻辑为：

```python
key_groups = source_keys[group_locs]
pooled = average_pool_qsa_keys(key_groups)
compressed_rope_positions = ...
normalized = self.normalize_compressed_keys(
    pooled, compressed_rope_positions
)
```

`average_pool_qsa_keys()` 在 FP32 中做平均，再转换回原始 dtype，以降低数值误差。

## 3.5 Block-level causal scoring

对于 query token $i$ 和 compressed block $b$，QSA 对所有 indexer query heads 的相似度做 ReLU 后求和：

$$
I_{i,b}
=
\sum_{h=1}^{H}
\operatorname{ReLU}
\left(
\langle q_{i,h},\bar k_b\rangle
\right)
$$

但只有已经完整出现的 block 才能被评分：

$$
 p_b+r-1\le i
$$

否则：

$$
I_{i,b}=-\infty
$$

这就是 block-level causal mask。当前尚未完成的 micro-block 不参与 compressed scoring，其已出现的 token 会作为 tail 单独处理。

## 3.6 Block top-k 与 token 展开

设 token budget 为 $K$，compression ratio 为 $r$，block budget 为：

$$
K_B=\left\lceil\frac K r\right\rceil
$$

Qwen3.8 的典型配置是：

$$
K=2048,\qquad r=4,\qquad K_B=512
$$

每个 query 首先选择 512 个重要 compressed blocks：

$$
B_i=\operatorname{TopK}_{K_B}(\{I_{i,b}\}_b)
$$

然后将每个 block 展开成其包含的原始 token。最终还要加入当前 incomplete block 中已经可见的 tail tokens：

$$
S_i=
\operatorname{Expand}(B_i)
\cup
\left\{
\left\lfloor\frac{i+1}{r}\right\rfloor r,
\ldots,i
\right\}
$$

因此最终 token index 数量最多为：

$$
K+r-1
$$

当 $K=2048,r=4$ 时，最终宽度为：

$$
2048+4-1=2051
$$

SGLang 中对应：

```python
block_indices = qsa_fast_topk(...)
selected = expand_qsa_block_indices(...)
```

`expand_qsa_block_indices()` 保证有效 token indices 连续排列，便于后续 FA2 packed path 使用。

## 3.7 Pending ring 的作用

decode 时，新 token 不一定刚好凑满一个 compression group，因此 QSA 为每个 request 维护一个小型 pending ring。

slot 计算方式为：

$$
\text{slot}
=
\text{request\_id}\times r + (\text{position}\bmod r)
$$

例如 $r=4$：

```text
position 0 → ring slot 0
position 1 → ring slot 1
position 2 → ring slot 2
position 3 → ring slot 3
```

当一个完整 group 形成后，四个 pending keys 被读取、平均池化并写入 compressed-key cache。

pending ring 的作用是：

- 避免每次 decode 重新扫描历史 token；
- 支持 compression group 边界；
- 让 CUDA Graph 可以使用固定形状 metadata；
- 让当前未完成 group 的 token 可以被单独保留。

## 3.8 Prefill 路径

prefill 中，每个 query row 对应序列中的一个位置。对序列长度 $L$，完整 compressed blocks 数量为：

$$
\left\lfloor\frac Lr\right\rfloor
$$

当前位置为 $p$ 时，可见完整 blocks 数量为：

$$
\min\left(
\left\lfloor\frac{p+1}{r}\right\rfloor,
\left\lfloor\frac Lr\right\rfloor
\right)
$$

SGLang 使用 `row_starts` 和 `row_ends` 描述每个 query row 在 packed compressed-key buffer 中可以访问的区间。

由于 prefill 可能产生非常大的 FP32 logits 矩阵，代码将 query rows 分块处理，并将 workspace 限制在约 128 MiB：

```python
_QSA_PREFILL_LOGITS_BUDGET_BYTES = 128 * 1024 * 1024
```

这样不会一次性构造过大的：

$$
[\text{query rows},\text{compressed keys}]
$$

矩阵。

## 3.9 Decode 路径

decode 使用 compressed page table：

```python
compressed_cache
compressed_page_table
compressed_lengths
```

compressed page table 可以根据 full KV page 和 compression ratio 进行算术转换。compressed cache 逻辑上相当于把原始 token page 按 block 粒度重新解释。

`compressed_decode_view()` 会：

1. 根据 request 的 token slot table 找出 page；
2. 将 full-token page 转为 compressed page；
3. 根据 `sequence_lengths // compress_ratio` 计算有效 compressed length；
4. 交给 decode MQA kernel 进行 block scoring。

## 3.10 Sparse Core Attention

得到 token indices 后，QSA 只加载选中的 token K/V，并执行真实 GQA。

核心 kernel 包括：

```python
_sparse_gqa_prefill
_sparse_gqa_chunk_prefill
```

计算形式仍然是标准 attention：

$$
O_h=
\sum_{j\in S_i}
\operatorname{softmax}
\left(
\frac{q_hk_j^\top}{\sqrt d}
\right)v_j
$$

不同点是可见集合从完整历史变为 $S_i$，其大小约为 2048。

对于 FA2 fallback，SGLang 会：

1. 统计每个 row 的有效 selected-token 数量；
2. 构造 `cu_seqlens_k`；
3. 将不规则的 K/V indices compact 到连续 buffer；
4. 调用 packed varlen attention。

对应函数包括：

```python
qwen_sparse_fa2_cu_seqlens_triton
qwen_sparse_kv_extraction_compact_triton
```

## 3.11 Speculative decoding 与 MTP

QSA 对 speculative decoding 做了特殊优化：

- target verify 可以重新运行 indexer；
- draft-extend 阶段可以捕获 target-aligned sparse indices；
- 后续 MTP decode steps 复用已捕获的 indices；
- 只追加从 captured length 到当前位置之间的新 tail。

`QSAMTPSharedSparseIndices` 保存每个 layer、每个 request 的 selection。后续查找时得到：

```text
[captured sparse indices] + [newly drafted tail tokens]
```

这样可以避免每一个 speculative decode step 重新执行：

- indexer Q/K projection；
- compressed MQA scoring；
- top-k selection。

报告显示，四步 speculative decoding 下，复用 QSA indices 不会明显降低 MTP accepted length。

## 3.12 QSA 的训练方法

QSA 不是从 dense model 直接切换到 sparse attention，而是经过两个 continued pretraining 阶段。

### Stage 1：Dense Distillation

首先使用 full attention backbone 作为 teacher。将 teacher 所有 attention heads 的 token-level attention probability 汇总并做 L1 normalization，得到：

$$
a_i\in\mathbb{R}^n
$$

然后使用 max pooling 将 token-level teacher distribution 对齐到 block 粒度：

$$
\bar a_{i,b}
=\operatorname{MaxPool}
(a_i[p_b:p_b+r-1])
$$

再做归一化：

$$
\hat a_i=rac{\bar a_i}{\|\bar a_i\|_1}
$$

之后用 KL loss 训练 indexer：

$$
\mathcal L_{KL}
=
\frac1N\sum_i
D_{KL}
\left(
\hat a_{i,:}
\middle\|
\operatorname{Softmax}(I_{i,:})
\right)
$$

使用 max pooling 而不是 average pooling，是为了避免重要 token 的信号在 block 聚合时被平均稀释。

报告中的配置：

- sequence length：256K；
- 8 条序列；
- 总计约 2B training tokens；
- indexer 单独训练 1000 steps；
- learning rate：$1\times10^{-3}$。

### Stage 2：Sparse Training

indexer 初始化后，整个 backbone 在 sparse attention 下继续训练：

1. indexer 选择 top-$K_B$ blocks；
2. blocks 展开为 token indices；
3. 加入当前 incomplete block 的 tail tokens；
4. core attention 只对这些 token 计算；
5. backbone 与 indexer 联合训练。

此阶段只在选中的 blocks 上计算 KL loss，并将 teacher probability 在选中 blocks 内重新归一化：

$$
\mathcal L_{KL}
=
\frac1N\sum_i
D_{KL}
\left(
\hat a_{i,B_i}
\middle\|
\operatorname{Softmax}(I_{i,B_i})
\right)
$$

最终 sparse CPT 阶段：

- 训练 8000 steps；
- learning rate：$2.5\times10^{-5}$；
- 96 条 256K 序列；
- 总计约 200B tokens。

报告指出，dense 初始化后直接启用 sparse attention 会造成明显性能下降；短暂的 joint sparse training 可以使 loss 恢复到接近 full-attention baseline。

## 3.13 QSA 的效率和效果

报告中的 kernel-level 结果，在 1M context 下约为：

- prefill attention module：7.6×；
- decode attention module：4.9×。

这些结果包含 indexer 和 sparse core attention，而不是只统计 sparse core kernel。

报告还显示：

- compression ratio 4 时，prefill indexer 约有 3.8× speedup；
- decode indexer 约有 4.4× speedup；
- QSA 从 64K context 开始出现明显收益；
- 上下文越长，收益越明显。

---

# 4. Gated Residual

## 4.1 为什么要扩展 residual stream

标准 Transformer 只有一条 residual stream：

$$
R'=R+y
$$

所有 block 都从同一条 stream 读取和写入。早期 block 的信息会逐渐与后续 block 的输出混合，后续层无法显式区分信息来源和时间尺度。

GR 将 residual stream 扩展为四条 branch：

$$
R=[R_1,R_2,R_3,R_4]
$$

每个 branch 的维度仍为 $d$，所以实际 widened residual 的形状是：

```text
[tokens, 4 * hidden_size]
```

逻辑形状是：

$$
R\in\mathbb{R}^{4\times d}
$$

对于 $T$ 个 token，则为：

$$
R\in\mathbb{R}^{T\times4d}
$$

## 4.2 GR 的整体流程

每个 attention 或 MLP 子层都执行：

```text
widened residual R
        │
        ├── GR mix
        │       ├── 每个 branch 独立 RMSNorm
        │       ├── 低秩 read gate
        │       └── 聚合成普通 hidden x
        │
        ├── 执行 attention / GDN / MLP
        │       └── y = F(x)
        │
        └── GR combine
                ├── 预测每个 branch 的 write scalar
                └── 将 y 写回所有 branch
```

数学上：

$$
R
\xrightarrow{\text{mix}}x
\xrightarrow{F}y
\xrightarrow{\text{combine}}R'
$$

其中：

- `mix` 是读取 residual；
- `combine` 是写回 residual。

## 4.3 第一步：branch-wise RMSNorm

对四个 branch 独立做 RMSNorm：

$$
\bar R_i=\operatorname{RMSNorm}(R_i;\gamma_i)
$$

其中：

- $i=1,\ldots,4$；
- 每个 branch 有独立 gain $\gamma_i\in\mathbb{R}^d$。

对应 SGLang 配置：

```python
hc_per_branch_norm=True
```

在 `GroupedGemmaRMSNorm` 中，group size 设置为 `hidden_size`，因此每个 branch 是独立的 normalization group。

这一步的作用是让不同 branch 在自己的尺度上参与后续 gate 预测，避免某个 branch 的大幅值影响其他 branch 的归一化。

## 4.4 第二步：low-rank read gate

将所有 normalized branches 拼接：

$$
\operatorname{vec}(\bar R)
=[\bar R_1;\bar R_2;\bar R_3;\bar R_4]
\in\mathbb{R}^{4d}
$$

然后经过 low-rank bottleneck：

$$
 z=
\operatorname{SiLU}
\left(
\frac14W_d\operatorname{vec}(\bar R)
\right)
$$

其中：

$$
W_d\in\mathbb{R}^{r\times4d}
$$

Qwen3.8 的 bottleneck rank 为 $r=320$。

再投影回 widened dimension：

$$
 u=W_u z
$$

$$
 W_u\in\mathbb{R}^{4d\times r}
$$

最后使用 sigmoid：

$$
 G=\operatorname{unvec}(\sigma(u))
\in\mathbb{R}^{4\times d}
$$

因此：

$$
G=[G_1,G_2,G_3,G_4]
$$

每个 $G_i$ 都是长度为 $d$ 的向量。

## 4.5 Read gate 是 elementwise gate

对于每个 token、每个 branch、每个 hidden channel，都有一个独立的 read gate：

```text
每个 token × 每个 branch × 每个 hidden channel
```

因此它不是每个 branch 一个 scalar，而是：

$$
G_i\in\mathbb{R}^d
$$

某个 channel 可以重点读取 branch 1，而另一个 channel 可能重点读取 branch 3。

这是 GR 相比简单的 branch scalar read 更强的原因。

## 4.6 第三步：生成 block 输入

四个 gated branches 做逐元素乘法并平均：

$$
 x
=\frac14\sum_{i=1}^{4}G_i\odot\bar R_i
$$

其中：

- $\odot$ 是逐元素乘法；
- $x\in\mathbb{R}^d$；
- $x$ 是 attention、GDN 或 MLP 的输入。

SGLang 中的核心逻辑为：

```python
input_mix_weight = F.silu(
    F.linear(hyper_input_normed, input_mix_weight_down) / hc
)
input_mix_weight = F.linear(input_mix_weight, input_mix_weight_up)
input_mix_weight = torch.sigmoid(input_mix_weight)
input_mix_weight = input_mix_weight.unflatten(-1, (hc, hs))

output = (
    input_mix_weight * hyper_input_normed.unflatten(-1, (hc, hs))
).mean(dim=-2)
```

伪代码：

```python
R_norm = [
    rmsnorm(R1),
    rmsnorm(R2),
    rmsnorm(R3),
    rmsnorm(R4),
]

G = sigmoid(
    W_up(
        silu(W_down(concat(R_norm)) / 4)
    )
)

x = mean(
    G[0] * R_norm[0],
    G[1] * R_norm[1],
    G[2] * R_norm[2],
    G[3] * R_norm[3],
)
```

## 4.7 `mix` 为什么返回 residual 信息

SGLang 中：

```python
hidden_states, residual = self.attn_hyper_connection.mix(hidden_states)
```

这里返回：

```text
mixed_input = x
residual_info = (原始 R, normalized R)
```

即：

$$
\operatorname{mix}(R)
\rightarrow
(x,(R,\bar R))
$$

后续 `combine()` 需要：

1. 原始 $R$，用于真正写回；
2. normalized $\bar R$，用于预测 write gate。

因此 `mix()` 不能只返回普通 hidden。

## 4.8 第四步：执行 block

得到普通 hidden $x$ 后，执行具体子层：

对于 QSA attention：

$$
 y_{attn}
=
\operatorname{o\_proj}
\left(
\operatorname{SparseAttention}(q,k,v)
\right)
$$

对于 GDN：

$$
 y_{gdn}=\operatorname{GDN}(x)
$$

对于 MLP/MoE：

$$
 y_{mlp}=\operatorname{MLP}(x)
$$

GR 的 `mix` 不负责 QSA 或 GDN，它只负责将 widened residual 读取成 block 输入。

## 4.9 第五步：生成 write gate

block 输出为：

$$
 y\in\mathbb{R}^d
$$

使用 normalized residual 预测每个 branch 的写入 gate：

$$
 s
=2\sigma
\left(
\frac14W_w\operatorname{vec}(\bar R)
\right)
$$

其中：

$$
 W_w\in\mathbb{R}^{4\times4d}
$$

并且：

$$
 s\in\mathbb{R}^{4}
$$

每个 token 有四个 scalar：

$$
 s=[s_1,s_2,s_3,s_4]
$$

由于有前面的系数 2：

$$
 s_i\in(0,2)
$$

## 4.10 Write gate 是 per-branch scalar

和 read gate 不同，write gate 不是逐 channel 的，而是：

```text
每个 token × 每个 branch 一个 scalar
```

写回公式为：

$$
R_i'=R_i+s_i y
$$

展开为：

$$
R_1'=R_1+s_1y
$$

$$
R_2'=R_2+s_2y
$$

$$
R_3'=R_3+s_3y
$$

$$
R_4'=R_4+s_4y
$$

同一个 $s_i$ 会乘到对应 branch 的全部 hidden channels。

对应代码：

```python
block_inject_weight_out = 2 * torch.sigmoid(
    F.linear(normed_residual, block_inject_weight) / hc
)

injection = (
    block_output.unsqueeze(-2)
    * block_inject_weight_out.unsqueeze(-1)
)

updated_residuals = R + injection
```

报告中的消融结论是：read gate 细化到 elementwise 很重要，但 write gate 细化到 channel 粒度几乎没有额外收益，因此最终采用 per-branch scalar write gate。

## 4.11 为什么使用 sigmoid

报告比较了 sigmoid 和 tanh 等 gate activation，最终选择 sigmoid，原因包括：

1. gate 输出为正；
2. gate 有界；
3. 不会像无约束线性权重一样无限放大 residual；
4. 训练稳定性更好；
5. loss 和 downstream benchmark 表现更好。

read gate：

$$
G_i\in(0,1)
$$

write gate：

$$
 s_i\in(0,2)
$$

这为 residual 的读写提供了受控的动态缩放。

## 4.12 GR 直接承担 pre-normalization

由于 GR read 已经包含：

1. branch-wise RMSNorm；
2. elementwise gated read；

因此 GR 不需要再额外放置一个普通 pre-norm。

普通结构是：

$$
 y=F(\operatorname{Norm}(x))
$$

而 GR 中可以理解为：

$$
 y=F(x)
$$

其中 $x$ 已经是经过 GR normalization 和 gating 后生成的输入。

因此 GR 不只是 residual wrapper，也承担了 block input normalization 的作用。

## 4.13 为什么去掉显式 branch mixing

一般 Hyper-Connections 还包含一个 branch mixing operator：

$$
H_{res}R
$$

用于显式地在 branches 之间交换信息。

但报告的消融显示，$H_{res}$ 的收益很小，代价却包括：

- 每个 block 额外读取完整 widened residual；
- 增加内存流量；
- 增加计算和实现复杂度；
- 引入额外稳定性问题。

因此 GR 去掉 $H_{res}$，保留：

- dynamic elementwise read；
- dynamic per-branch scalar write；
- branch-wise normalization。

branch 之间的信息交换通过以下方式完成：

1. 所有 branches 共同决定 read gate；
2. block output 可以按不同强度写入所有 branches；
3. 下一个 block 再通过新的 read gate 选择各 branch 的内容。

所以虽然没有显式的：

$$
R_i\leftarrow\sum_j a_{ij}R_j
$$

但仍然实现了动态信息路由。

## 4.14 Branch 如何保存跨层信息

由于没有 $H_{res}$，branch $c$ 在 block $v$ 之前可以展开为：

$$
R_c^{(v)}
=
R_c^{(0)}+
\sum_{u<v}s_c^{(u)}y^{(u)}
$$

其中：

- $R_c^{(0)}$ 是初始 embedding；
- $y^{(u)}$ 是第 $u$ 个 block 的输出；
- $s_c^{(u)}$ 是该 block 写入 branch $c$ 的 gate。

因此不同 branch 可以自然形成不同用途：

- 一条 branch 保留较早层的长程信息；
- 其他 branch 保存局部更新；
- 后续 attention/QSA 层通过 read gate 重新选择这些信息。

在读取层 $v$，贡献为：

$$
 x^{(v)}
=
\frac14\sum_c
G_c^{(v)}
\odot
\operatorname{RMSNorm}(R_c^{(v)})
$$

报告中的 branch contribution analysis 发现，一条 branch 往往承载跨越很多层的早期 attention/GDN 信息，而另外几条 branch 更偏局部路径。长距离路径主要会被后续的 full-attention/QSA 层读取。

## 4.15 Attention 和 MLP 使用独立 GR

Qwen3.8 的每个 decoder layer 具有两套独立 GR：

```python
self.attn_hyper_connection = GatedResidual(...)
self.mlp_hyper_connection = GatedResidual(...)
```

Attention 子层：

```text
R
 │
 ├── attn_gr.mix(R) → x_attn
 │
 ├── QSA / GDN / attention
 │
 └── attn_gr.combine(y_attn, R_info) → R_attn
```

数学形式：

$$
 x_{attn}=\operatorname{GR}_{attn}^{mix}(R)
$$

$$
 y_{attn}=F_{attn}(x_{attn})
$$

$$
 R_{attn}=\operatorname{GR}_{attn}^{combine}(R,y_{attn})
$$

MLP 子层：

```text
R_attn
 │
 ├── mlp_gr.mix(R_attn) → x_mlp
 │
 ├── MLP / MoE
 │
 └── mlp_gr.combine(y_mlp, R_info) → R_next
```

数学形式：

$$
 x_{mlp}=\operatorname{GR}_{mlp}^{mix}(R_{attn})
$$

$$
 y_{mlp}=F_{mlp}(x_{mlp})
$$

$$
 R_{next}=\operatorname{GR}_{mlp}^{combine}(R_{attn},y_{mlp})
$$

因此 attention 和 MLP 分别学习自己的：

- branch-wise normalization；
- low-rank read gate；
- per-branch write gate。

## 4.16 在 `Qwen4Exp` 中的调用顺序

`Qwen4ExpAttentionDecoderLayer.forward()` 的逻辑可以简化为：

```python
hidden_states, residual = self._prepare_qwen4_exp_attn(
    hidden_states,
    residual,
    forward_batch,
)
```

其中会执行：

```python
hidden_states, residual = self.attn_hyper_connection.mix(
    hidden_states
)
```

之后执行 QSA 或其他 attention：

```python
hidden_states = self.self_attention(...)
```

然后进入 MLP 前，先将 attention output 写回 attention branches，再从新的 branches 读取 MLP input：

```python
hidden_states = self.attn_hyper_connection.combine(
    hidden_states,
    residual,
)

hidden_states, residual = self.mlp_hyper_connection.mix(
    hidden_states
)
```

最后 MLP output 再写回：

```python
hidden_states = self.mlp_hyper_connection.combine(
    hidden_states,
    residual,
)
```

单层完整流程为：

```text
输入 R
 │
 ├── Attention GR mix
 │
 ├── Attention / QSA / GDN
 │
 ├── Attention GR combine
 │
 ├── MLP GR mix
 │
 ├── MLP / MoE
 │
 └── MLP GR combine
       │
       ▼
下一层的 R
```

## 4.17 GR 与普通 residual add 的区别

| 操作 | 普通 residual | GR |
|---|---|---|
| residual 数量 | 1 | 4 |
| read | 固定读取 | 逐元素动态 gate |
| write | 固定加法 | 每 branch 动态 scalar |
| normalization | 普通 pre-norm | 集成在 GR read 中 |
| branch mixing | 单 stream 隐式混合 | 无显式 $H_{res}$ |
| block 输入 | 单一 stream | 四路 gated aggregation |
| residual 形状 | $d$ | $4d$ |

## 4.18 最终模型输出

所有 decoder layers 执行结束后，模型内部仍然是 widened residual：

```text
[tokens, 4 * hidden_size]
```

模型末端使用一个只包含 `mix` 的 GR：

```python
self.hyper_connection_mixer = GatedResidual(
    hc_config,
    use_combine=False,
)
```

最终：

$$
 h_{final}
=
\frac14\sum_{i=1}^{4}
G_i\odot\operatorname{RMSNorm}(R_i)
$$

将其恢复为：

```text
[tokens, hidden_size]
```

然后交给 logits head 或 multimodal output head。

## 4.19 GR 的 fused kernel 优化

`mix` 包含多个操作：

1. low-rank down projection；
2. SiLU；
3. low-rank up projection；
4. sigmoid；
5. branch-wise multiply；
6. branch reduction。

decode 小 batch 下，每个操作单独启动 kernel 会带来较高 launch overhead。因此 SGLang 在 `hc_mix_triton.py` 中实现了 persistent fused kernel，将这些操作合并：

```text
down projection
      ↓
SiLU
      ↓
up projection
      ↓
sigmoid
      ↓
branch-wise gated reduction
```

支持条件包括：

- CUDA；
- FP16/BF16；
- 小 batch，最多 16 rows；
- widened width 可被 2048 整除；
- contiguous tensor；
- 非 deterministic inference。

`combine` 也有对应 CUDA kernel，将：

$$
R_i+s_i y
$$

融合执行，减少中间 tensor 和内存读写。

## 4.20 Deterministic inference

persistent mix kernel 的 down projection 使用 device-scope atomic accumulation。不同 CTA 的累加顺序可能不同，因此结果可能存在微小浮点差异。

开启 deterministic inference 时，SGLang 会禁用该 atomic-based fused path，回退到确定性实现。

这是：

```text
普通推理：优先 fused kernel，降低 decode latency
确定性推理：关闭 atomic fused kernel，保证结果可复现
```

之间的取舍。

---

# 5. QSA 与 GR 的协同

QSA 和 GR 解决的是不同层面的问题。

## QSA 的职责：决定看哪些 token

QSA：

- 对压缩后的历史做快速检索；
- 选择重要 blocks；
- 展开为 token indices；
- 对选中 token 执行精确 sparse attention。

它主要优化 attention 的上下文访问成本。

## GDN 的职责：递归压缩历史

GDN 层：

- 递归更新状态；
- 以线性复杂度处理历史；
- 保存连续状态形式的记忆。

## GR 的职责：管理跨层 residual information flow

GR：

- 保存多条 residual streams；
- 动态决定每条 branch 如何被读取；
- 动态决定 block 输出写回哪些 branch；
- 在模型末端将 widened stream 聚合回普通 hidden。

一次 QSA attention layer 中，实际流程是：

```text
四路 widened residual
          │
          └── GR read
                  │
                  ├── indexer Q/K projection
                  │       ├── compressed key
                  │       ├── block scoring
                  │       └── token selection
                  │
                  ├── 主 attention Q/K/V projection
                  │
                  ├── sparse GQA
                  │
                  ├── attention output gate
                  │
                  ├── o_proj
                  │
                  └── GR write
                          │
                    写回四路 residual
```

因此 QSA 产生的 attention output 不会直接覆盖 residual，而是依次经过：

1. attention 自身的 output gate；
2. `o_proj`；
3. GR 的 branch-wise write gate；
4. 写回四条 residual branches。

可以总结为：

```text
GDN：recurrent memory
QSA：sparse precise retrieval
GR ：multi-branch residual routing
```

---

# 6. 报告中的关键消融结论

## 6.1 QSA

1. micro-block compression 比跨层共享 indices 更适合 GDN/attention hybrid 架构；
2. compression ratio 越大，indexer 成本越低，但过度压缩会损失检索质量；
3. 4 个 indexer query heads 已经足够；
4. dense 初始化后直接启用 QSA 会导致性能下降；
5. 短暂 joint sparse training 可以恢复性能；
6. QSA 在长上下文下的收益不断扩大；
7. MTP 复用 QSA indices 不会显著降低 accepted length。

## 6.2 Gated Residual

1. 仅扩大 residual stream 就能降低 loss；
2. dynamic read/write 优于 static read/write；
3. read gate 的 elementwise 粒度非常重要；
4. write gate 使用 per-branch scalar 就足够；
5. gate 应该根据所有 branches 共同预测；
6. 每个 branch 独立 RMSNorm 有额外收益；
7. 显式 $H_{res}$ branch mixing 几乎没有收益；
8. 去掉 $H_{res}$ 可以降低 widened residual 的内存流量；
9. sigmoid gate 带来更好的训练稳定性和下游表现。

---

# 7. 最终总结

## QSA

QSA 的本质是：

$$
\boxed{
\text{compressed block retrieval}
+
\text{original-token precise attention}
}
$$

它先对 compressed blocks 做低成本 index attention，再将重要 blocks 展开为原始 token，最后只对这些原始 K/V 做 sparse GQA/FlashAttention。

## Gated Residual

GR 的本质是：

$$
\boxed{
\text{four-way widened residual}
+
\text{elementwise dynamic read}
+
\text{per-branch scalar write}
}
$$

完整公式为：

$$
\begin{aligned}
\bar R_i &= \operatorname{RMSNorm}(R_i) \\
G &= \operatorname{unvec}\left[
\sigma\left(
W_u\operatorname{SiLU}
\left(\frac14W_d\operatorname{vec}(\bar R)\right)
\right)
\right] \\
x &= \frac14\sum_i G_i\odot\bar R_i \\
y &= F(x) \\
s &= 2\sigma\left(\frac14W_w\operatorname{vec}(\bar R)\right) \\
R_i' &= R_i+s_i y
\end{aligned}
$$

## 二者的协同

- GDN 负责 recurrent compression；
- QSA 负责 sparse long-context retrieval；
- GR 负责跨层、多分支 residual routing；
- QSA 的 attention output 先经过 attention output gate，再由 GR write gate 写回多路 residual；
- 最终 GR read 将四路 widened state 聚合回普通 hidden。

因此，Qwen3.8-Flash-Next 的核心设计可以概括为：

> 用 GDN 低成本维护连续记忆，用 QSA 保留精确的长上下文 token 检索能力，再用 GR 将 residual capacity 扩展为多条可动态路由的信息通道。

---

# 8. Hugging Face Transformers `qwen4_exp` 实现补充

本节根据以下文件对上面的论文和 SGLang 分析进行实现级校准：

- `/home/ov2022/workspace/transformers/src/transformers/models/qwen4_exp/configuration_qwen4_exp.py`
- `/home/ov2022/workspace/transformers/src/transformers/models/qwen4_exp/modular_qwen4_exp.py`
- `/home/ov2022/workspace/transformers/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py`

其中 `modeling_qwen4_exp.py` 是由 `modular_qwen4_exp.py` 生成的文件，后续代码修改应优先修改 modular 文件。

## 8.1 架构开关与层布局

`Qwen4ExpTextConfig.__post_init__()` 在没有显式提供 `layer_types` 时，按 `full_attention_interval=4` 生成层类型：

```text
第 1、2、3 层：linear_attention（Gated DeltaNet）
第 4 层：qwen_sparse_attention（QSA）
第 5、6、7 层：linear_attention
第 8 层：qwen_sparse_attention
...
```

因此“每四层一个全局 attention”在 Transformers 中已经具体化为“每四层一个 QSA 层”，而不是普通 dense full attention。若 checkpoint 的 `layer_types` 仍包含 `full_attention`，配置初始化会把它转换为 `qwen_sparse_attention`。

QSA 的五个配置字段必须同时存在：

| 字段 | 作用 |
|---|---|
| `indexer_n_heads` | indexer query head 数 |
| `indexer_kv_heads` | indexer key head 数；实现强制为 1 |
| `indexer_head_dim` | indexer Q/K head dimension |
| `indexer_budget` | 每个 query 从完整 block 中选择的 token 上限 |
| `indexer_compress_ratio` | 每个 compressed block 包含的连续 token 数 |

配置还强制 `indexer_budget % indexer_compress_ratio == 0`，并检查 RoPE 的实际 rotary dimension 不超过 `indexer_head_dim`。这保证了 `block_topk = indexer_budget // indexer_compress_ratio` 是整数。

## 8.2 Transformers 中 QSA indexer 的精确数据流

`Qwen4ExpTextQSAIndexer` 与主 attention 使用独立的 `index_qk_proj`：

$$
W_{index}:d\rightarrow (H_q+H_k) d_i
$$

线性层输出沿最后一维切分为 query 和 token key：

$$
q\in\mathbb{R}^{B\times T\times H_q\times d_i},\qquad
k_{token}\in\mathbb{R}^{B\times T\times d_i}
$$

query 先经过 RMSNorm，再应用当前位置的 RoPE；raw token key 暂不加 RoPE，也不在 token 粒度先做 key normalization。历史 raw key 在启用 cache 时通过 `past_key_values.update_indexer()` 保存。

对每个 batch 和 query 位置，indexer 从主 causal mask 中提取可见 token indices。随后：

1. 按可见 token 的顺序切分完整 block；
2. 每个 block 对 raw key 做 FP32 average pooling；
3. 对 pooled key 做 RMSNorm；
4. 将 RoPE 应用到 block 的**首个 token 位置**；
5. 计算所有 indexer query heads 的相似度。

第 $b$ 个 block 的 pooled key 为：

$$
\bar k_b=\operatorname{RMSNorm}\left(
\frac{1}{r}\sum_{j=0}^{r-1}k_{br+j}
\right)
$$

再以 $br$ 作为该 block 的 rotary position。query 与 block key 的评分实现为：

$$
I_{i,b}=\frac{1}{\sqrt{d_i}}
\sum_{h=1}^{H_q}\operatorname{ReLU}
\left(q_{i,h}^{\mathsf T}\bar k_b\right)
$$

这里的 `/ sqrt(indexer_head_dim)` 位于 ReLU 求和之后，等价于对总分缩放，而不是分别缩放每个 head 后再改变排序。

### 完整 block、top-k 与 tail

对于当前 query 的可见 token 数 $V_i$，完整 block 数为：

$$
n_{complete}=\left\lfloor\frac{V_i}{r}\right\rfloor
$$

indexer 只对这 $n_{complete}$ 个 block 做 `topk(min(block_topk, n_complete))`。未完成 block 的 token 不参加评分，也不占用 block top-k 名额，但会原样追加到 selected token 列表：

$$
S_i=\operatorname{Expand}(\operatorname{TopK}(I_{i,:}))
\;\Vert\;
\operatorname{tail}(V_i\bmod r)
$$

因此预分配的 selected index 宽度是：

$$
\mathrm{indexer\_budget}+r-1
$$

其中前一部分来自最多 `block_topk` 个完整 block，后 $r-1$ 个位置用于当前未完成 block。长度不足的槽位用 `-1` 填充。

实现并不直接返回 index 数组，而是把 selected indices scatter 成一个布尔/加性 attention mask。所有 `-1` 会先 scatter 到额外的哨兵列，再裁掉该列，从而避免无效 index 参与主 attention。

## 8.3 QSA 主 attention：mask 筛选而非第二套 attention 算子

`Qwen4ExpTextAttention.forward()` 的执行顺序是：

```text
indexer(hidden_states, causal_mask)
    → selected_token_mask
    → 与 causal_mask 合并
    → 主 q/k/v projection
    → Qwen3.5 风格 attention interface
    → attention output gate
    → o_proj
```

主 attention 的投影与 indexer 完全分开：

- `q_proj` 输出 `2 * num_attention_heads * head_dim`，一半是 query，一半是 output gate；
- `k_proj` 和 `v_proj` 使用 `num_key_value_heads`；
- query/key 各自做 RMSNorm，再应用 RoPE；
- cache 保存的是主 attention 的 key/value。

indexer 产生的 mask 与 causal mask 通过逻辑与（bool mask）或加法（float mask）合并。也就是说，HF eager/SDPA 路径的 QSA 本质是：

> 使用 indexer 生成稀疏可见集合，再调用标准 attention 算子完成真实 Q/K/V 计算。

主 attention 输出还会经过独立的 elementwise output gate：

$$
g=\operatorname{sigmoid}(g_{q\_proj}),\qquad
y_{attn}=W_o\left(g\odot\operatorname{Attention}(q,k,v)\right)
$$

因此不能把这个 gate 与 GR 的 write gate 混为一谈：前者调节 attention 输出的 hidden channel，后者调节该输出写入哪一条 residual branch。

## 8.4 Transformers 中 Gated Residual 的真实实现

`Qwen4ExpTextGatedResidual` 的输入维度是：

$$
d_{hc}=C\cdot d
$$

默认 $C=4$。`Qwen4ExpTextRMSNorm(group_size=hidden_size)` 会把最后一维 reshape 为 `[..., hc_count, hidden_size]`，所以四条 branch 各自独立归一化，而不是对完整的 $4d$ 向量做一次 RMSNorm。

需要注意 RMSNorm 的参数化：

$$
\operatorname{RMSNorm}(x)=
x\left(\operatorname{mean}(x^2)+\epsilon\right)^{-1/2}
\odot(1+w)
$$

模型初始化时将 $w$ 置零，因此初始时 gain 为 1。

### Read：低秩、逐元素、四路平均

设独立归一化后的 branch 为 $\bar R_c$，拼接为 $\bar R\in\mathbb{R}^{Cd}$。代码中的 read gate 为：

$$
G=\operatorname{unvec}\left(
\operatorname{sigmoid}\left[
W_{up}\operatorname{SiLU}\left(\frac{W_{down}\bar R}{C}\right)
\right]
\right)
$$

其中：

$$
W_{down}\in\mathbb{R}^{r\times Cd},\qquad
W_{up}\in\mathbb{R}^{Cd\times r},\qquad r=\text{hc\_lowrank}
$$

默认 `hc_lowrank=320`。$G_c\in(0,1)^d$ 是每个 branch、每个 channel 的 gate，最终输入为：

$$
x=\frac{1}{C}\sum_{c=1}^{C}G_c\odot\bar R_c
$$

这里的除以 $C$ 是显式的 branch mean，不是通过 gate normalization 间接实现的。

### Write：每 branch 一个 scalar

当 GR 用于 attention 或 MLP 子层时，额外创建：

$$
W_{inject}\in\mathbb{R}^{C\times Cd}
$$

并根据同一个 `hyper_input_normed` 产生：

$$
s=2\operatorname{sigmoid}\left(\frac{W_{inject}\bar R}{C}\right)
\in(0,2)^C
$$

若 block 输出为 $y\in\mathbb{R}^d$，则写回为：

$$
R_c'=R_c+s_cy
$$

write gate 是每个 token、每个 branch 一个 scalar，并非每个 channel 一个权重。`Qwen4ExpTextGatedResidual.forward()` 返回三项：

```text
(mixed_input, 原始 hyper_input, injection_weights)
```

因此 HF 实现没有单独的 `mix()` / `combine()` 方法；同一个 `forward` 在 `use_combine=True` 时一次性生成 read 结果、保留原始 residual 并计算 write weights。

## 8.5 Decoder layer 中的 GR 时序

`Qwen4ExpTextDecoderLayer.forward()` 的实际流程如下：

```text
可选 PLE 注入到四路 residual
    ↓
attn_hyper_connection：R → x_attn
    ↓
QSA 或 Gated DeltaNet
    ↓
R_attn = R + s_attn · y_attn
    ↓
mlp_hyper_connection：R_attn → x_mlp
    ↓
Sparse MoE
    ↓
R_next = R_attn + s_mlp · y_mlp
```

模型入口把普通 token embedding 复制到四个 branch：

$$
R^{(0)}=\operatorname{repeat}(E, C)
$$

这不是把 hidden size 改成四倍后再用一个新的 embedding 投影，而是同一个 embedding 向量的四份初始副本。每层的 attention GR 与 MLP GR 参数独立，因此两类子层拥有不同的 read/write routing。

模型末尾的 `hyper_connection_mixer` 使用 `use_combine=False`，只执行 read：

$$
h_{final}=\frac{1}{C}\sum_{c=1}^{C}G_c\odot\operatorname{RMSNorm}(R_c)
$$

它把 `[batch, sequence, C * hidden_size]` 恢复成 `[batch, sequence, hidden_size]`，之后才能送入 lm head。

## 8.6 与 SGLang 描述的对应和差异

两套实现的核心数学结构一致：独立 indexer、压缩 block、top-k 展开、tail 保留，以及四路 GR 的 elementwise read/per-branch scalar write。实现层面需要区分：

| 方面 | Transformers `qwen4_exp` | SGLang 路径 |
|---|---|---|
| QSA 输出 | 生成 selected-token attention mask | 生成 packed token indices 并调用 sparse kernels |
| compressed key | forward 中按可见 token 动态 pooling | 有 compressed cache、page table 和 pending ring 优化 |
| prefill/decode | 统一的 eager/SDPA mask 逻辑 | 针对 prefill/decode、FA2、CUDA Graph 分别优化 |
| GR 执行 | Python/PyTorch 模块，读写在 layer 内完成 | 另有 fused mix/combine Triton/CUDA 路径 |
| speculative/MTP | 本文件未实现 SGLang 的共享 sparse-index 机制 | 支持 captured indices 与 tail 增量复用 |

所以，HF 文件适合验证模型语义、张量形状和 checkpoint 参数含义；SGLang 文件则在相同语义上进一步实现了 page-based KV、稀疏 kernel、pending ring 和 speculative decoding 优化。不能把 SGLang 的优化数据结构反推成 HF reference implementation 必然存在的状态。

## 8.7 基于代码的最终归纳

QSA 在 `qwen4_exp` 中可以精确概括为：

$$
\boxed{
    \mathrm{独立\ indexer\ Q/K}
\rightarrow
    \mathrm{FP32\ block\ pooling + block\ top-k}
\rightarrow
    \mathrm{tail\ 保留}
\rightarrow
    \mathrm{selected\text{-}token\ mask}
\rightarrow
    \mathrm{标准\ GQA}
}
$$

Gated Residual 可以精确概括为：

$$
\boxed{
    \mathrm{四路独立\ RMSNorm}
\rightarrow
    \mathrm{低秩逐元素\ read\ gate}
\rightarrow
    \mathrm{四路平均形成\ block\ input}
\rightarrow
    \mathrm{per\text{-}branch\ scalar\ write\ gate}
}
$$

最容易混淆的三个 gate 应区分如下：

1. QSA 主 attention 的 `q_proj` gate：逐 channel 调节 attention 输出；
2. GR read gate：逐 branch、逐 channel 选择 residual 内容；
3. GR write gate：逐 branch、单 scalar 控制 block 输出写回强度。

三者串联后，QSA 层的输出路径是：

$$
R
\xrightarrow{\mathrm{GR\ read}}
x
\xrightarrow{\mathrm{QSA\ index+GQA}}
\xrightarrow{\mathrm{attention\ output\ gate}+o_{proj}}
y
\xrightarrow{\mathrm{GR\ write}}
R'
$$

这说明 QSA 负责“从哪些历史 token 读取”，而 GR 负责“从哪些 residual branch 读取以及把结果写回哪些 branch”；它们是 token 级稀疏检索与 layer 级 residual 路由的两个正交控制面。

---

# 9. PLE：Per-Layer Embedding 的注入机制

## 9.1 PLE 的定位

PLE（Per-Layer Embedding）是 Qwen4-Exp 中额外的 token n-gram 特征通道。它不是普通的输入 embedding，也不是 attention/GDN 的 cache memory，而是只在指定 decoder layer 生成的、面向该层的 lexical feature residual。

配置通过 `ple_layer_ids` 指定启用 PLE 的层号，层号是从 1 开始计数的 decoder layer id。配置验证还要求：

- `ple_layer_ids` 非空时，所有 PLE 层必须是 `linear_attention` 层；
- `ple_embed_dim` 必须能被 n-gram head 总数整除；
- 必须提供 `eos_token_id`，用于 n-gram 边界处理。

所以当前实现中，PLE 不会直接挂在 QSA 层上；它只出现在 Gated DeltaNet 层所在的指定位置。

## 9.2 PLE 的输入：hashed n-gram embedding

在 PLE 层中，`Qwen4ExpTextNGramEmbedding` 根据原始 token ids 构造 n-gram 特征。默认 `ngram_size=3`，因此会生成：

- 2-gram 特征；
- 3-gram 特征。

每种 n-gram 使用 `heads_per_ngram` 个独立 hash heads。默认值为 8，因此总 head 数为：

$$
H_{ngram}=(ngram\_size-1)\times heads\_per\_ngram
$$

默认情况下：

$$
H_{ngram}=(3-1)\times8=16
$$

每个 n-gram head 使用独立的素数词表大小和 offset，避免不同 head 的 embedding 行发生直接重叠。n-gram id 通过 token id 的 layer-specific 奇数乘数和 bitwise XOR 构造，再对该 head 的素数词表大小取模：

$$
id_{ngram,h}=\left(\bigoplus_j
(token_{i-j}\times m_j)\right)\bmod V_h
$$

其中 $m_j$ 由 `seed` 和 PLE 层索引确定，因此不同 PLE 层使用不同的 hash multiplier。所有 head 的 embedding 向量拼接后形成：

$$
e_i^{ngram}\in\mathbb{R}^{ple\_embed\_dim}
$$

这部分 embedding 规模很大；代码注释说明完整 n-gram embedding 约为几十 GiB，因此模型把它标记为 `_no_placement_params`，避免自动 device map 因尝试整体 offload 而导致显存/内存问题。

## 9.3 EOS 与增量 decode 的 n-gram 上下文

PLE 不会跨越 EOS 拼接 n-gram。`_shift_right_ignore_eos()` 在右移 token 序列时，如果历史窗口跨过 EOS，就用 `eos_token_id` 填充，而不是取前一个 segment 的 token。

在使用 cache 时，PLE 将最近的 `ngram_size - 1` 个 token 保存到当前 layer 的：

```text
conv_states[state_idx=2]
```

下一次 decode 会读取这段 previous context，与新输入 token 拼接后重新计算 n-gram。首次输入不足 context length 时显式使用 EOS 左填充；这与普通 convolution state 的零填充不同。

因此 PLE 的 token 历史是独立维护的，不能用 GDN 的 recurrent state 或 PLE convolution state 替代。

## 9.4 PLE 如何生成 widened residual 增量

设 n-gram embedding 为 $e_i^{ngram}$，当前 decoder residual 为：

$$
R_i\in\mathbb{R}^{C d}
$$

其中 $C=hc\_count$，默认 $C=4$。PLE 先生成：

### Key：每个 residual branch 一组 key

$$
k_i=\operatorname{RMSNorm}_{branch}
\left(W_k e_i^{ngram}\right)
\in\mathbb{R}^{Cd}
$$

reshape 后得到 $k_{i,c}\in\mathbb{R}^{d}$，每个 branch 有一组 key。

### Value：所有 branch 共享一个 value

$$
v_i=W_v e_i^{ngram}\in\mathbb{R}^{d}
$$

PLE 只生成一个 hidden-size value，而不是每个 branch 生成一份独立 value。这个共享 value 随后由各 branch 的 gate 以不同强度复制。

### Query：来自当前 widened residual

$$
q_{i,c}=\operatorname{RMSNorm}_{branch}(R_{i,c})\in\mathbb{R}^{d}
$$

注意这里的 query 不是主 attention 的 Q，也不是 QSA indexer 的 Q；它是 PLE 用于判断“当前 residual 是否需要这组 n-gram feature”的 branch-wise residual query。

## 9.5 Branch gate 与局部卷积

对每个 token、每个 residual branch，PLE 计算 key-query 点积：

$$
a_{i,c}=\frac{k_{i,c}^{\mathsf T}q_{i,c}}{\sqrt d}
$$

代码随后执行 signed square-root 变换：

$$
\widetilde{a}_{i,c}=\operatorname{sign}(a_{i,c})\sqrt{\max(|a_{i,c}|,10^{-6})}
$$

再使用 sigmoid 得到 gate：

$$
g_{i,c}=\sigma(\tilde a_{i,c})\in(0,1)
$$

由于 $v_i$ 对所有 branch 共享，初始的 gated value 为：

$$
u_{i,c}=g_{i,c}v_i
$$

拼接所有 branch 后，PLE 对 gated value 做 branch-wise RMSNorm，并经过一个 depthwise dilated Conv1d。默认参数为：

- `ple_conv_kernel_size=4`；
- dilation=`ngram_size`，默认为 3；
- 每个 channel 独立卷积（`groups=hc_count * hidden_size`）。

最终 PLE 输出为直接项与局部上下文项之和：

$$
PLE(R_i,e_i)=u_i+\operatorname{Conv}_{depthwise,dilation=ngram\_size}
\left(\operatorname{RMSNorm}(u_i)\right)
$$

输出形状保持为：

$$
[B,T,Cd]
$$

这说明 PLE 不改变 residual 的 widened dimension，也不把 n-gram embedding 直接送入 attention 或 MLP。

## 9.6 PLE 注入到哪里

在 `Qwen4ExpTextDecoderLayer.forward()` 中，PLE 的调用位于该 layer 的最开始：

```text
hidden_states = hidden_states + self.ple(...)
```

随后才执行：

```text
hidden_states, hyper_input, injection_weights = attn_hyper_connection(hidden_states)
```

因此准确的数据流是：

```text
原始 widened residual R
    │
    ├── PLE(R, token_ids) → ΔPLE，形状 [B,T,C·d]
    │
    └── R_ple = R + ΔPLE
        │
        └── Attention/GDN GR read
            │
            ├── QSA（若该层是 sparse attention）
            └── GDN（当前 PLE 层只能是 linear attention）
```

换句话说，PLE 注入的是：

> 当前 decoder layer 的 widened residual stream，在 attention/GDN 子层的 GR read 之前。

它不是：

- 注入到主 attention 的 Q/K/V；
- 注入到 QSA compressed key；
- 注入到 GDN recurrent state；
- 注入到 MLP 输出之后；
- 直接替换 token embedding。

## 9.7 PLE 与运行时模块的关联

### 与 `Qwen4ExpTextModel` 的关联

模型入口负责提供 PLE 所需的 token ids：

- 如果使用 `input_ids`，直接把它们传入 `ple_input_ids`；
- 如果只使用 `inputs_embeds` 且启用了 PLE，则调用 `reverse_embedding()` 从 embedding 反推 token ids；
- 因此 PLE 依赖离散 token ids，即使主模型路径使用的是 `inputs_embeds`。

模型入口还为普通 token embedding 复制出 $C$ 个 residual branch：

$$
R^{(0)}=\operatorname{repeat}(E,C)
$$

PLE 输出与这个 widened residual 直接相加，所以 PLE 的 `key_proj` 和 `value_proj` 必须分别适配 `ple_embed_dim` 到 `Cd` 与 `d`。

### 与 GR 的关联

PLE 输出先加到 residual，再被 `attn_hyper_connection` 读取。因此 GR 决定 PLE 特征最终如何进入 attention/GDN：

- GR 的 branch-wise RMSNorm 会重新归一化 PLE 注入后的各 branch；
- GR read gate 逐 channel 决定 PLE 增量对 block input 的影响；
- block 输出再通过 GR write gate 写回四个 branch。

PLE 本身有一套 key/query gate，但这套 gate 只控制 n-gram value 是否注入各 branch，不能与 GR read/write gate 混淆。

### 与 GDN 的关联

配置强制 PLE 只位于 `linear_attention` 层，因此 PLE 的直接消费者通常是同一层的 Gated DeltaNet：

$$
R\xrightarrow{+PLE}R_{ple}
\xrightarrow{GR\ read}x
\xrightarrow{GDN}y
\xrightarrow{GR\ write}R'
$$

但 PLE 不写入 GDN 的 recurrent matrix state。GDN 仍然使用自己的 `conv_states[state_idx=0]` 和 `recurrent_states[0]`。

### 与 QSA 的关联

当前配置校验不允许 PLE 位于 `qwen_sparse_attention` 层，所以 PLE 不会直接改变 QSA indexer 的输入。只有在更高层的 residual 传递后，PLE 所在层产生的更新才会间接影响后续 QSA 层的 hidden states。

### 与 Cache 的关联

启用 PLE 后，每个相关 layer 的 cache 至少需要三个彼此独立的状态：

| state | 用途 |
|---|---|
| `state_idx=0` | GDN 的 causal convolution state |
| `state_idx=1` | PLE 输出的 dilated depthwise convolution state |
| `state_idx=2` | PLE n-gram 最近 token context |

没有 PLE 时，配置将 `number_of_conv_states` 设为 1，只保留 GDN convolution state。该设计避免了三种不同语义的历史状态互相覆盖。

### 与 mask/padding 的关联

PLE 接收 `conv_mask`，也就是 linear-attention/recurrent mask 路径产生的 mask：

1. `gated_value` 和其 normalized 版本先将 padding 位置置零；
2. 再进入 PLE 的 dilated convolution；
3. 因此 padding token 不应向后续位置传播 lexical feature。

在模型入口，如果启用了 PLE，`ple_input_ids` 也会在 recurrent mask 对应的无效位置被替换为 EOS，进一步阻止 n-gram 跨越 padding 区域。

## 9.8 一层中的完整顺序

对于启用 PLE 的 linear-attention layer，完整顺序是：

```text
input_ids
    │
    ├── hashed 2/3-gram embedding
    ├── key/value projection
    ├── residual-query gate
    ├── depthwise dilated convolution
    └── ΔPLE [B,T,C·d]
        │
        ▼
R [B,T,C·d] + ΔPLE
        │
        ▼
Attention GR read
        │
        ▼
Gated DeltaNet
        │
        ▼
Attention GR write
        │
        ▼
MLP GR read
        │
        ▼
Sparse MoE
        │
        ▼
MLP GR write
```

最终可以这样理解：

> PLE 是一种按层启用的、基于 hashed n-gram 的局部 lexical residual 注入器。它在 GDN/QSA/MLP 之前并不直接改变这些模块的内部输入接口，而是先修改 widened residual；随后由 attention GR 统一决定这些 n-gram 特征有多少进入当前 block，并由 GR write 将 block 结果重新分配到多路 residual。

---

# 10. llama.cpp 中 Qwen4-Exp QSA 的实现审计与优化

本节补充对当前 `/home/ov2022/workspace/qsa/llama.cpp` 工作树的代码级分析。主要参考：

- `src/models/qwen4exp.cpp`
- `src/llama-memory-hybrid-idx.h`
- `src/llama-memory-hybrid-idx.cpp`
- `src/llama-kv-cache.cpp`
- `src/llama-model.cpp`

需要区分目标算法和当前实现：前文描述的是 QSA 的模型语义以及 SGLang 的高性能路径；当前 llama.cpp 已实现 indexer、block grouping、pooling、score、tail 和 top-k，但主 attention 仍然是 dense MHA 加 mask。因此当前路径更准确地称为：

```text
QSA-aware dense attention / masked-QSA prototype
```

它还不是完整的 compact block-sparse attention。

## 10.1 当前数据流

Qwen4-Exp 使用三类状态：

```text
attention KV cache：主 attention 的 K/V
recurrent cache：GDN/linear-attention 状态
indexer KV cache：每个 token 一个 raw indexer K
```

`llama_memory_hybrid_idx` 在 attention/recurrent cache 之外创建 indexer cache。indexer cache 是单 key head，并与 attention cache 共用 cell slot layout，但数据 buffer 独立。

`build_layer_attn()` 对 full-attention layer 的 QSA 路径大致为：

```text
GR mixed hidden `cur`
       │
       ├── index_k_proj → raw indexer K → indexer cache
       ├── index_q_proj → indexer Q
       ├── raw K gather → block mean pooling
       ├── pooled K RMSNorm → block-start RoPE
       ├── indexer Q × pooled block K
       │       → ReLU → head reduction → block score
       ├── block score 映射到 token score
       │       → causal/sequence/tail bias → top-k
       ├── 主 q/k/v projection → 主 attention KV cache
       └── dense MHA + selected-token mask
```

indexer Q/K 与主 attention Q/K/V 是两套独立投影。indexer 只负责检索，主 Q/K/V 负责最终 attention。

## 10.2 raw indexer K 和 block metadata

当前 indexer cache 保存的是 raw、未归一化、未旋转的 K。cache 构造时显式禁用内部 RoPE，实际顺序是：

```text
raw K 写入 cache
→ block pooling
→ pooled K RMSNorm
→ block-start RoPE
```

`set_input_qsa()` 生成四类 metadata：

| metadata | 形状 | 作用 |
|---|---|---|
| `cell_blk` | `[n_kv, n_stream]` | 每个 cache cell 所属的逻辑 block |
| `blk_cells` | `[ratio*n_blocks, n_stream]` | block 成员到 cache cell 的 gather 映射 |
| `blk_pos` | `[4*n_blocks*n_stream]` | block 首 token 的 RoPE/mRoPE 位置 |
| `bias` | block-level 或 cell-level | causal、sequence、tail 可见性 |

QSA block 不是物理 PA page。实际分组键包含：

```text
(sequence identity, position bucket)
```

只有含有完整 `ratio` 个位置槽的 group 才能生成 pooled block。未完成的 group 作为 tail 处理，并通过大正 bias 保证可见。二维 mRoPE 场景还可能需要对重复 temporal position 的 cell 做 rank 排序，不能假设 cache cell 按位置连续。

## 10.3 当前并没有持久化 block summary

这是当前实现和高性能 QSA 设计之间最重要的差异。

在 `build_qsa_top_k()` 中，当前 graph 每次都会重新执行：

```text
get raw indexer K
→ ggml_get_rows(blk_cells)
→ ratio 次 slice/add
→ scale(1/ratio)
→ index_k_norm
→ block-start RoPE
```

得到的 `pooled`/`indexer_k` 是当前 graph 的临时 tensor，并没有保存为“每个完整 logical block 一个 summary”的持久化状态。

因此当前 decode 会重复处理历史 block；每个 QSA layer 也会用自己的 `index_k_proj/index_k_norm` 重新生成 block 表示。可以共享 block membership 和 cache 生命周期，但不能直接跨 layer 共享 summary 数值。

应区分：

```text
raw indexer K cache：当前实现的持久化源数据
pooled block K：当前 forward 的临时中间结果
block summary cache：建议新增的派生状态
```

## 10.4 当前 top-k 的真实语义

当前 score 先是 block-level：

$$
S_{b,t,s}=\sum_h \operatorname{ReLU}\left(q_{h,t,s}^{T}k_{b,s}\right)
$$

然后通过 `cell_blk` 将 block score 扩展回 token：

```text
[n_blocks, n_query, n_stream]
    → [n_kv, n_query, n_stream]
```

最后在 token score 上执行：

```text
width = min(n_kv, indexer_top_k + ratio - 1)
top_k = ggml_top_k(expanded_score, width)
```

所以当前不是严格的：

```text
block score → block top-k → block expansion
```

而是：

```text
block score
→ 复制到 block 内 token
→ 加 causal/sequence/tail bias
→ token-level top-k
```

`indexer_top_k + ratio - 1` 用于覆盖完整 block 以及最多 `ratio-1` 个 tail token。将其直接改为 block-level top-k 前，需要验证 tie、tail、多 sequence 和二维位置下的逐 token 等价性。

## 10.5 当前主 attention 仍然是 dense

`build_attn_qsa()` 先写入当前主 K/V，然后构造完整的 `-INF` mask，用 `ggml_set_rows()` 将 top-k 对应位置恢复为可见，最后调用普通 `build_attn_mha()`。

实际计算仍接近：

```text
Q × K_all^T
→ full score/softmax shape
→ mask non-selected positions
→ full V path
```

因此当前 QSA 没有真正消除：

- 主 K cache 的完整读取；
- 主 V cache 的完整读取；
- dense QK 的完整扫描；
- 大型 dense mask 的创建和搬运。

源码中的 `TODO: enable sparse attention when we are ready` 也明确说明了这一点。当前 QSA 的 GPU 收益不能按“最终 attention 只读取 top-k token”估算，它仍然是 masked dense attention。

## 10.6 Prefill、decode 和 CPU 瓶颈

### Prefill

prefill 的主要成本为：

1. 每个 QSA layer 重新 gather/pool 历史 indexer K；
2. query rows 与 block summary 的 score；
3. block score 扩展到完整 `n_kv`；
4. 在 token width 上执行 top-k；
5. 创建 dense top-k mask；
6. 继续执行 dense MHA。

### Decode

decode 通常为 `n_tokens=1`，多序列 decode 通过 `n_stream` 表示。当前没有专门的 compact sparse decode graph，仍会重复 indexer 处理并执行 dense attention。

另一个明确瓶颈是 `llama_memory_hybrid_idx::set_input_qsa()`：每个 ubatch 都扫描完整 `n_kv`，重建 `cell_blk`、`blk_cells`、`blk_pos` 和 bias。单 token decode 只新增一个 token，却可能重新处理整个历史 cache metadata。

## 10.7 优化路线

### P0：实现 compact selected-KV attention

在保持当前 top-k 语义不变的前提下，先将：

```text
top-k token indices → dense mask → dense MHA
```

改为：

```text
top-k token indices → gather selected K/V → attention over selected length
```

decode 优先实现 single-query/multi-sequence sparse GQA，prefill 再实现 packed multi-query sparse attention。这样才会真正减少主 K/V 的带宽和 QK/V 计算量。

### P1：持久化完整 block summary

建议为每个 layer、sequence、logical block 保存：

```text
summary[b, Di]
valid[b]
logical_start[b]
logical_end[b]
```

更新策略为：

```text
新 token → raw K cache
block 未完成 → pending state
block 完成 → 只生成一次 summary
```

rollback、sequence copy/remove、position shift 必须同步更新 summary validity。summary 应按 logical sequence/block 管理，而不是简单绑定物理 cell。

### P1：缓存和增量更新 host metadata

给 `set_input_qsa()` 增加 layout generation/version。只有 slot allocation、sequence copy/remove、rollback、position shift、mRoPE rank 关系或 SWA window 变化时才做全量重建。普通 decode 只更新当前 sequence 的 tail 和最后一个 block。

### P1：decode 使用 partition + finalization

长上下文 decode 建议使用：

```text
score_partition
    → local top-k
    → topk_finalization
    → selected block expansion
    → tail append
```

partition 只沿 summary/block range 切分，不沿 indexer head `Hidx` 切分。沿 `Hidx` 切分会重复加载相同 summary，并增加带宽和归约成本。

### P2：融合 indexer kernel

可融合：

```text
FP32 pooling + RMSNorm + block RoPE
dot product + ReLU + head reduction + bias
```

这样可以减少中间 tensor、graph split 和显存往返。

### P2：block-level top-k

在验证语义等价后，再把：

```text
block score → token expansion → token top-k
```

改为：

```text
block score → block top-k → selected block expansion → tail append
```

这会同时减少 score expansion 和大宽度 top-k。

## 10.8 推荐的 Q0/Q1/Q2/Q3 kernel 边界

面向 OpenVINO CM/Xe2/Xe3 的实现，建议将 llama.cpp 当前混合在 `build_attn_qsa()` 中的逻辑拆成：

```text
Q0 qsa_kv_cache_update
    current main K/V → main PA KV cache

Q1 qsa_prepare
    hidden → indexer Q/raw-K/pending/complete summaries

Q2 qsa_score_topk
    summary × indexer Q → selected block metadata

Q3 qsa_sparse_attention
    selected block metadata → main PA KV gather → sparse attention
```

依赖关系：

```text
Q0 ───────────────┐
                  ├── Q3
Q1 → Q2 ──────────┘
```

Q0 与 Q1 可以独立调度；Q3 必须等待主 KV 写入和 Q2 selected metadata。`o_proj`、主 attention 的 output gate 与 GR write gate 都应保持为独立语义。

## 10.9 验证建议

至少比较三条路径：

```text
dense attention
current masked-QSA
compact sparse-QSA
```

测试维度应覆盖：

```text
context：4K / 8K / 16K / 32K / 64K
mode：prefill / decode
ratio：模型实际 ratio 及边界值
batch：single sequence / multi-stream
position：普通文本 / 2D mRoPE
cache：append / rollback / sequence copy
```

应分别统计 `set_input_qsa()`、raw-K 写入、pooling、indexer score、top-k、mask、selected K/V gather、Q3 attention 和总 layer latency。正确性上需要逐项比较 raw K、pooled block K、score、tail indices、top-k indices、attention output，以及 rollback/multi-sequence 后的 cache 状态。

## 10.10 最终判断

当前 llama.cpp 的 QSA 逻辑闭环是：

```text
raw indexer K
  → block grouping
  → mean pooling
  → norm/RoPE
  → indexer score
  → tail-aware top-k
  → dense selected-token mask
  → dense main attention
```

高性能 QSA 尚缺少两个关键条件：

1. 主 attention 仍是 dense masked attention，而不是 compact sparse attention；
2. block summary 是每次 forward 临时生成，而不是完整 block 完成后的持久化派生状态。

因此推荐的实施顺序是：

```text
先实现 Q3 compact sparse attention
→ 缓存/增量更新 host metadata
→ 持久化 Q1 block summary
→ 引入 block-level partition/finalization 和融合 kernel
```

这样可以分别验证 GPU 主 attention、CPU metadata、summary lifecycle 和 top-k 语义，避免一次性改变过多边界后难以定位性能或正确性回归。

# 11. SGLang 当前 Qwen4-Exp QSA 实现审计与优化

本节基于当前工作树 `/home/ov2022/workspace/qsa/sglang` 的 `main` 分支源码，重点分析 compressed QSA，而不是 Qwen3.5/DeepSeek 的 tokenwise DSA。当前实现与前面的 llama.cpp 审计有本质差异：

```text
llama.cpp：QSA 选择 + dense masked attention 原型
SGLang：独立 indexer + 持久化 compressed-K + compact sparse GQA/FA
```

## 11.1 总体调用链

Qwen4-Exp 是 hybrid model。full-attention layer 使用 QSA，linear-attention layer 仍走 GDN；GR/HyperConnection 位于 QSA 之外，负责 `[B,T,4D]` 与 `[B,T,D]` 的分支读写。QSA indexer 不直接处理 GR，也不替代主 attention 的 Q/K/V projection。

`Qwen4ExpAttentionDecoderLayer` 的主要路径是：

```text
hidden_states [M,D]
    ├─ 主 attention Q/K/V + output gate
    ├─ QSAIndexer：独立 index_qk_proj、压缩 cache、block top-k
    └─ QwenSparseAttnBackend：logical token → physical slot → sparse GQA
```

indexer 的投影输出为：

```text
q       [M, Nq, Di]
token_k [M, 1,  Di]
```

当前 compressed QSA 强制 `indexer_kv_heads == 1`，并要求：

```text
indexer_budget % compress_ratio == 0
indexer_budget / compress_ratio ∈ {512, 2048}
```

设压缩比为 $R$，token budget 为 $K_t$，则实际先选择 $K_b=K_t/R$ 个 compressed blocks，再展开为最多 `K_t + R - 1` 个 token index。这个设计避免了先将 block 分数复制成 token 分数再做一次 token-level top-k。

## 11.2 Indexer 的 Q/K 语义与压缩时机

`QSAIndexer.project_qk()` 使用独立的 `index_qk_proj`：

1. indexer Q 经过 `GemmaRMSNorm` 和 Qwen4-Exp 自身的 mRoPE；
2. indexer K 保留为 raw、未归一化、未旋转的 `token_k`；
3. raw K 首先写入 per-request pending ring；
4. 一个完整的 $R$ token group 到齐后，再执行 FP32 mean pooling、K RMSNorm 和 group-start mRoPE。

因此 compressed key 的语义是：

$$
K_c=
\operatorname{RoPE}_{p_g}\left(
    \operatorname{RMSNorm}
    \left(\frac{1}{R}\sum_{i=0}^{R-1}K_{gR+i}^{raw}\right)
\right)
$$

这点与“先逐 token RoPE 再平均”不同，且与本报告前面确定的 block-summary 语义一致。CUDA 条件满足时，`qsa_index_q_norm_rope_store` 融合 Q 的 norm/RoPE 和 raw K 写入；`qsa_index_k_compress_store` 融合 mean、norm、mRoPE 和 compressed-cache store。

## 11.3 KV pool：持久化 compressed K 与 pending ring

`QSATokenToKVPool` 没有单独的 compressed-cache allocator；它借用 full-KV allocator 的 slot 生命周期：

```text
full slot       = req_to_token / full KV allocator 管理
compressed slot = full slot // R
compressed page = full page / R
```

压缩 QSA 的关键缓冲区为：

```text
qsa_key_state_buffer_pool:
        [num_request_slots * R, 1, Di]，每个 request 只保留未完成 group

qsa_rope_position_buffer:
        [num_request_slots * R, 3]，保存 pending raw K 的 mRoPE 坐标

qsa_compressed_k_buffer_pool:
        [num_layers, compressed_capacity, 1, Di]，持久化完整 group 的 compressed K
```

pending slot 为：

$$
slot=req\_pool\_idx\times R+(position\bmod R)
$$

compressed page 的正确性依赖 `page_size % R == 0`，因此一个 compression group 不会跨 full-KV page。当前 Qwen4-Exp override 将 compressed QSA 的 page size 设为 64；这不是随意的调参，而是为了使 `full_slot // R` 和 page-aligned compressed addressing 成立。

需要区分两个概念：SGLang **有持久化的 compressed K cache**，但没有另外维护一个独立的、按 block summary 对象管理的 metadata cache。摘要内容就在 compressed-K pool 中；有效范围、page table 和待压缩 group 的状态由每次 forward metadata 重建或 CUDA Graph buffer 更新。

## 11.4 Prefill/extend 的 block scoring

`QSAIndexer.select_prefill_tokens()` 先从 compressed-K pool 按 request 拼接可见 block，`build_qsa_row_ranges()` 计算每个 query row 的 packed range：

```text
compressed_length = floor(sequence_length / R)
row_start         = request 的 compressed prefix 起点
row_end           = row_start + min(floor((query_position + 1) / R), compressed_length)
```

然后使用 MQA index score：

$$
s_{m,b}=\frac{1}{\sqrt{D_i}}
\sum_h\operatorname{ReLU}(q_{m,h}\cdot k_{b})
$$

CUDA 上的 `qsa_mqa_prefill()` 使用 TileLang GEMM；当前实现仍产生 `[rows, compressed_keys]` 的 FP32 logits，但按 128 MiB workspace 上限分块 row。随后使用 `fast_topk`/`fast_topk_v2` 选择 $K_b$ 个 block，并由 Triton `expand_qsa_block_indices` 展开 token indices。

展开逻辑有两个重要语义：

1. 一个 selected block 展开为连续的 $R$ 个 token；
2. 当前不完整 tail 不依赖 summary，直接追加到输出，因此 tail 始终可见。

无 prefix 时，主 attention 使用 `sparse_gqa_fwd_interface_triton`，直接在当前 packed K/V 上按 selected indices 做在线 softmax；因此不是 dense mask。存在 prefix 时，当前实现仍会把完整 prefix K/V 从 paged cache `index_select` 到连续的 `k_parts/v_parts`，再调用 `sparse_gqa_fwd_interface_triton_ck`，这是 prefill 的主要可优化点。

## 11.5 Decode、page table 与 sparse GQA

decode/target-verify/draft-extend 都采用“一 query token 一 metadata row”的布局。`QSAIndexer.forward_cuda()` 的顺序是：

```text
index_qk_proj
    → raw K 写 pending ring
    → 完整 group 写 compressed K
    → compressed page table + compressed length
    → MQA block score
    → block top-k
    → token index expansion
```

`compressed_decode_view()` 从 full-KV `token_slot_table` 读取 page id，并将有效长度设为 `floor(sequence_length / R)`。超出有效长度的 page-table entry 可以是 stale，但不会被读取；因此长度保护是 cache 正确性的关键。

主 attention 收到的是 logical token indices。`QwenSparseAttnBackend._logical_to_physical()` 再通过：

```text
token_to_batch_idx
    + sequence_lengths
    + token_slot_table
    → physical KV slots
```

在 CUDA 上，当前实现不是直接让 FA kernel 读取任意 paged logical index，而是先执行：

```text
qwen_sparse_valid_counts_triton
    → qwen_sparse_kv_extraction_compact_triton
    → FlashInfer TRTLLM decode 或 FlashAttention varlen
```

SM100/SM120 优先使用 FlashInfer TRTLLM paged decode；其他 GPU 使用 FA2/FA4 varlen fallback。两条路径都读取 compact 后的 selected K/V，attention 计算量接近 $O(B\times K_t\times H_q\times D)$，而不是 $O(B\times L\times H_q\times D)$。

## 11.6 CUDA Graph 与 metadata 优化

compressed QSA 的 CUDA Graph 路径已经把很多 host 工作搬到了 GPU。`graph_metadata.py` 的两个 Triton kernel 根据 `seq_lens`、request index 和 `req_to_token` 重建：

```text
row layout
compressed lengths
compressed write locations
pending ring slots
group ring locations
compressed page table
logical positions
```

这避免 decode replay 中频繁的 D2H sequence-length readback。capture 阶段分配固定地址 buffer，replay 阶段只更新其内容；request slot 0 作为 inert/dump row 处理 padding 和 graph warmup，避免无效行污染真实 pending ring。

但该优化主要覆盖 `QSATokenToKVPool` compressed variant。当前 tokenwise QSA 仍要求关闭 CUDA Graph，因为它没有稳定的 compressed graph metadata 方案。这也说明 OpenVINO 实现应将“常规 eager metadata”和“固定形状 graph metadata”分开设计，而不应把 page table 构造隐含在主 attention kernel中。

## 11.7 MTP/speculative 的索引复用

Qwen4-Exp MTP 使用单个 full-attention layer 的 MTP model，并保留 HC hidden-state 传递。QSA 的 `QSAMTPSharedSparseIndices` 在 draft-extend 中捕获每个 request 的最后有效 row；后续 draft decode 不再重新运行 indexer，而是复用 frozen selection，并追加从 capture position 之后新增的连续 tail。

当前限制是明确且重要的：

```text
speculative_eagle_topk == 1
draft_token_num <= compress_ratio
不支持 adaptive speculation 的通用树状索引复用
```

原因是 pending ring 按 `position % R` 存储，一个 verify window 超过 $R$ 会在同一 forward 内发生 ring slot collision。该限制不是性能实现细节，而是当前 state layout 的语义边界。

## 11.8 已实现优化与主要瓶颈

### 已经实现的优化

- indexer Q norm、mRoPE、raw-K store 的 fused CUDA kernel；
- mean、K norm、group-start mRoPE、compressed-K store 的 fused kernel；
- compressed block-level top-k，避免先生成 token-level score；
- Triton block-index expansion，并保证 valid token index 连续；
- prefill sparse GQA，在线维护 softmax，而不是 dense mask；
- decode selected-K/V compact 后接 TRTLLM/FA varlen；
- CUDA Graph GPU metadata rebuild，减少 host synchronization；
- MTP draft-extend 的 index reuse 与 tail append；
- Qwen4-Exp decode 小 token 数时使用 alt stream 与 attention overlap indexer，当前阈值为 1024 tokens。

### 主要瓶颈

1. prefill score 仍物化 `[rows, compressed_keys]` FP32 logits；
2. decode MQA 仍输出 `[batch, max_model_len]` logits，而不是在 kernel 内直接维护 top-k；
3. decode 仍有 valid-count、logical-to-physical/KV compact、最终 FA/TRTLLM 三段流水；
4. prefix prefill 仍完整重打包 prefix K/V；
5. block expansion、valid-count 和 compact 都是额外 kernel；
6. pending ring 的 mRoPE 坐标使用 int64，且 QSA metadata eager 路径仍会创建多个临时 tensor；
7. MTP index reuse 只覆盖严格 chain speculation。

## 11.9 对 OpenVINO CM Q0/Q1/Q2/Q3 的映射

当前 SGLang 代码可以直接映射到此前确定的四类 OpenVINO kernel，但边界应保持清晰：

| OpenVINO kernel | SGLang 对应职责 | CM 实现重点 |
|---|---|---|
| Q0 `qsa_kv_cache_update` | 主 K/V 写 full PA cache；SGLang 的 `set_kv_buffer` | 与 indexer raw K/compressed K 完全解耦 |
| Q1 `qsa_prepare` | `project_qk`、pending ring、完整 group compression | fused Q norm/RoPE/store；mean→norm→group-start RoPE；按 request ring 维护未完成 group |
| Q2 `qsa_score_topk` | `qsa_mqa_prefill/decode`、`qsa_fast_topk`、block expansion | 先 block top-k；decode 优先按 summary/page range 分区和 finalization，禁止沿 `Hidx` 分区 |
| Q3 `qsa_sparse_attention` | `sparse_gqa_fwd_interface_triton`、KV compact 后 FA/TRTLLM | 直接按 logical→physical selected slots 做 GQA；不要退化为 dense mask |

对 Xe2/Xe3，建议优先实现 Q2 的在线 block top-k 和 Q3 的直接 paged selected-load。SGLang 当前的“先 compact K/V、再调用通用 FA”适合 CUDA 生态，但在 Intel GPU 上会暴露额外带宽和 kernel-launch成本。CM 版本可以将 valid-count、logical-to-physical、selected K/V load 与 Q3 融合，避免中间 compact buffer。

## 11.10 最终判断

当前 SGLang Qwen4-Exp QSA 已经是完整的 sparse execution path：

```text
独立 indexer Q/K
    → raw-K pending ring
    → 持久化 compressed-K cache
    → compressed block MQA score
    → block top-k + tail expansion
    → logical-to-physical selected KV
    → sparse GQA / FlashAttention / TRTLLM decode
```

因此 OpenVINO 移植不应复刻 llama.cpp 的 dense masked attention，也不应把 block summary 每次 forward 全量重算。正确的优先级是：

```text
Q0 主 KV 更新与 Q1 indexer/cache 更新并行设计
→ Q2 block score + online top-k + tail
→ Q3 直接 paged selected-load sparse GQA
→ graph-stable metadata 与 MTP index reuse
```

其中 compressed K 是持久化派生 cache，pending ring 只保存尚未完成的 compression group；两者都必须纳入 cache lifecycle、rollback、prefix sharing 和多序列隔离的正确性测试。

# 12. OpenVINO CM 中 QSA 与 XAttention 的 Sparse Attention 对比

本节对比当前 OpenVINO CM 实现中的 QSA sparse attention 与 XAttention + Paged Attention。主要参考：

- `src/plugins/intel_gpu/src/graph/impls/cm/qsa/qsa_sparse_attention.cm`
- `src/plugins/intel_gpu/src/graph/impls/cm/qsa/qsa_sparse_attention_dpas.cm`
- `src/plugins/intel_gpu/src/graph/impls/cm/qsa/qsa_score_topk_fused.cm`
- `src/plugins/intel_gpu/src/graph/impls/cm/xattn_gemm_qk.cm`
- `src/plugins/intel_gpu/src/graph/impls/cm/xattn_find_block.cm`
- `src/plugins/intel_gpu/src/graph/impls/cm/xattn_post_proc.cm`
- `src/plugins/intel_gpu/src/graph/impls/cm/pa_multi_token.cm`
- `src/plugins/intel_gpu/src/graph/impls/cm/include/cm_pa_xe2.hpp`

## 12.1 总体执行路径

QSA 和 XAttention 都是 block-sparse attention，但二者的稀疏选择和执行组织方式不同：

```text
QSA:
indexer Q × compressed summary K
    → per-query block top-k
    → selected block list
    → direct sparse QK / online softmax / PV

XAttention:
main Q × KV block estimate
    → query-block sparse mask
    → workgroup mask merge
    → Paged Attention block iteration
    → masked QK / online softmax / PV
```

## 12.2 详细对比表

| 对比维度 | QSA sparse attention | XAttention + Paged Attention |
|---|---|---|
| 选择入口 | 独立 indexer Q 与 compressed summary K | 主 attention Q 与 KV block 做 XAttention estimate |
| 选择 kernel | `qsa_score_topk_fused.cm`、decode score/finalization | `xattn_gemm_qk.cm` → `xattn_find_block.cm` |
| 选择后处理 | 直接写入 `selected_blocks` 和 `selected_counts` | `xattn_post_proc.cm` 将多个 query block mask OR 合并 |
| 选择结果 | `selected_blocks[token, topk]` | `sparse_block_mask` 与 `sparse_block_mask_wg` |
| 选择结构 | 紧凑的升序 block ID 列表 | block mask，通常按 head/query-block/key-block 索引 |
| query 选择粒度 | 每个 query token 一组 selected blocks | 初始为 query block/tile 相关，最终可能提升到 workgroup union |
| query 是否共享逻辑选择 | 默认不共享；每个 token 独立 top-k | 初始不完全共享；post-process 后多个 query block 可能共享并集 mask |
| main attention head 关系 | indexer reduction 后，同一 token 的所有 Hq heads 共享该 token 的 selected set | mask 通常带 attention-head 维度，不能默认所有 head 共享 |
| KV 选择粒度 | QSA compressed block，展开后通常包含 `r` 个连续 token | XAttention sparse block，常见 block size 为 1、128 或 256 |
| 选择数量 | 最多固定 `QSA_BLOCK_TOPK` 个 block | 由阈值/累计分数和 causal 范围决定，不保证固定 top-k |
| 是否严格 token top-k | 先 block top-k，再展开为 token；不是直接对 raw token 独立评分 | 主要是 block mask，不是 token-level compact top-k |
| compressed K | 使用 QSA compressed summary cache | 不依赖 QSA summary cache，estimate 直接服务于 sparse block mask |
| 主 attention K/V | 原始 PA K/V cache | 原始 PA K/V cache |
| 主 QK 计算 | 只对 selected block load 的 K 做计算 | Paged Attention 遍历 mask 允许的 KV block |
| attention 实现 | `qsa_sparse_attention.cm` 逐 selected block 计算 | `pa_multi_token.cm`、`pa_small_q.cm` 的 PA tile pipeline |
| DPAS/矩阵化路径 | `qsa_sparse_attention_dpas.cm` | `cm_pa_xe2.hpp` 中的 CM/LSC 优化 PA 路径 |
| softmax | FP32 online softmax | PA 分块 online softmax |
| masked score | selected block 对应列保留，其余列置为负无穷 | sparse block mask 与 causal mask 共同限制可见列 |
| KV 加载方式 | selected block 直接 gather；每个 block 的 `r` 行可连续加载 | 按 PA tile/block 搬运到 SLM/register，再复用给 workgroup 内 query |
| 多 query KV 复用 | reference 路径较少；DPAS 路径对多个 token 的 selected list 取 union | workgroup-level mask 和 SLM staging 原生强调多 query 复用 |
| union 的用途 | 仅为填充 DPAS N 维度和减少重复 KV load | 为 workgroup 级 block skip/load 和 KV 搬运复用 |
| union 是否改变逻辑选择 | 不改变；`step_masks` 对每个 query column 再单独屏蔽 | merged mask 可能扩大物理执行域，fine mask/causal mask 负责最终限制 |
| causal tail | incomplete QSA block 作为 always-visible tail 单独处理 | 通常由 PA 的 `kv_stop` 和 row-level causal mask 处理 |
| invalid slot 处理 | `-1` 在地址计算前跳过 | 通过 sparse mask/边界条件跳过 block 或 token |
| PA page 映射 | selected block → logical token → PA page/token offset | block mask → PA block iteration → PA page/token offset |
| 中间 compact K/V | 通常不需要，selected block 直接在 Q3 load | 通常不需要单独 compact list，PA 直接按 mask 遍历 |
| metadata 规模 | `O(num_tokens × topk)` | `O(heads × query_blocks × key_blocks)`，具体取决于 mask 布局 |
| 稀疏率控制 | 直接由 block top-k 控制 | 由 score threshold、mask 和 block size 间接控制 |
| 高稀疏率优势 | 选择列表短，直接跳过大量 KV | 大 block skip 和 workgroup KV reuse 效果明显 |
| 低稀疏率/选择密集时 | selected list 变长，gather 和 list merge 成本增加 | PA tile/SLM 复用更容易摊薄 estimate 成本 |
| 典型优化瓶颈 | top-k、selected list merge、非规则 gather | estimate scratch、mask merge、workgroup load 范围过宽 |

## 12.3 QSA sparse attention 的计算粒度

### Reference kernel

`qsa_sparse_attention.cm` 的 dispatch 是：

```text
gws = [QSA_HEADS, num_tokens, 1]
```

即一个 kernel instance 主要处理一个 `(attention head, query token)`。它对该 query 的 selected blocks 逐个执行：

```text
selected block
    → PA page/token offset
    → 连续加载 r 个 K token 和 r 个 V token
    → QK
    → online softmax
    → PV 累加
```

这里的 selected set 是 per-query-token 的。由于 QSA indexer heads 已经在 Q2 中完成 reduction，同一 query token 的所有 main attention heads 共享相同的 selected block list；但不同 query token 通常拥有不同 list。

### DPAS kernel

`qsa_sparse_attention_dpas.cm` 不把 16 个 query token 直接当作同一选择集合处理，而是把 DPAS 的 N 维度组织成 `(head, token)` pair：

```text
QSA_HEADS_PER_TILE × QSA_TOKENS_PER_TILE = 16
```

同一个 tile 内多个 token 的 selected list 会做升序多路 merge，形成 union block list。随后 `step_masks` 对每个 token column 重新施加选择：

```text
KV row 对 query column 可见，当且仅当该 column 所属 token 选择了该 block
```

所以 QSA DPAS 的准确语义是：

```text
物理 KV load：union of several token selections
逻辑 attention：每个 token 仍只看自己的 selected blocks
```

这与 XAttention 的 workgroup union mask 不同。QSA DPAS 通过 per-column mask 保证 union 只是一种执行优化，不改变每个 query 的稀疏集合。

## 12.4 XAttention sparse attention 的计算粒度

XAttention 由三个 estimate 阶段组成：

```text
xattn_gemm_qk
    → xattn_find_block
    → xattn_post_proc
```

`xattn_gemm_qk.cm` 的 scratch layout 保留了 query 维度，例如：

```text
kq_max_wg:          [batch, head, query_group, q_stride]
kq_exp_partial_sum: [batch, head, q_stride, k_block]
```

这表明 estimate 并不是为整个序列生成一个全局共享的 KV 分数向量。

`xattn_find_block.cm` 调用 `find()` 生成 block mask，`find_block.hpp` 使用类似以下的 query-block 偏移：

```cpp
block_mask += m_block * k_block_pad;
```

因此初始 block selection 至少保留 query-block 依赖。不同 query block 可以产生不同的 KV block mask。

随后 `xattn_post_proc.cm` 对多个 query block 的 mask 做 OR：

```cpp
new_mask |= cur_mask;
```

输出的 `merged_block_mask` 是 workgroup 执行级别的并集，而不是证明所有 query 的原始 score 或 top-k 都相同。

## 12.5 逻辑选择与物理 KV 加载的区别

这是比较 QSA 和 XAttention 时最重要的区分：

| 层次 | QSA | XAttention |
|---|---|---|
| 逻辑 selection | 每个 query token 产生 selected block list | query block/tile 先产生 block mask |
| selection merge | DPAS tile 内取 union，但用 per-column mask 恢复语义 | post-process 将多个 query block mask OR 成 workgroup mask |
| 实际 load domain | union list 中的 block | workgroup merged mask 允许的 block |
| 单个 query 最终可见性 | selected block mask + tail | fine mask + causal mask |
| 额外加载 block 是否必然影响输出 | 不影响，显式 per-column 屏蔽 | 正确的 PA mask/causal 路径应屏蔽不属于该 query 的位置 |

在 `cm_pa_xe2.hpp` 的优化路径中，满足以下条件时会启用固定 workgroup 的 block-sparse pipeline：

```text
SPARSE_BLOCK_SIZE == CMPA_WG_SEQ_LEN
SPARSE_BLOCK_SIZE ∈ {128, 256}
```

此时 `skip_load(kv_blk)` 按 sparse block 决定整个 workgroup 是否跳过该 KV block。这样多个 query token 可以共享：

- sparse block 的 load/skip 决策；
- SLM 中的 K/V staging；
- 一部分 KV 遍历和 causal 上界。

但 `pa_small_q.cm` 也说明，tile 可以用较大的共同 KV 迭代范围，再对每个 query row 重新施加更严格的 causal mask。因此：

```text
共享 KV 加载范围 ≠ 每个 query 拥有相同的逻辑可见集合
```

## 12.6 稀疏 attention 伪代码对照

### QSA reference

```text
for each query token t:
    q = main_query[t]
    selected = selected_blocks[t]
    for each selected QSA block b:
        Kb, Vb = load_pa_block(b)
        online_softmax_update(q, Kb, Vb)
    attend_to_visible_tail(t)
    write_output(t)
```

### QSA DPAS

```text
for each (head-tile, token-tile):
    union = merge(selected_blocks[token_0], ..., selected_blocks[token_T])
    for each block b in union:
        Kb, Vb = load_once_for_tile(b)
        St = Q_tile × Kb
        apply_per_token_selection_mask(St, b)
        online_softmax_and_PV(St, Vb)
```

### XAttention + PA

```text
for each query block/workgroup:
    estimate = qk_block_estimate(query_tile, kv_blocks)
    fine_mask = find_sparse_blocks(estimate)
    wg_mask = OR_masks_for_workgroup(fine_mask)

    for each KV block in workgroup iteration:
        if wg_mask says skip:
            continue
        stage_KV_in_SLM()
        for each query row:
            apply_fine_mask_and_causal_mask()
            online_softmax_and_PV()
```

## 12.7 对 OpenVINO CM 设计的启示

如果目标是实现此前规划的 QSA Q0–Q3，不能直接把 XAttention 的 merged workgroup mask 当作 QSA 的 `selected_blocks`：

1. QSA Q2 应保持 per-query block top-k 语义；
2. QSA Q3 可以仿照 XAttention 的 workgroup/SLM reuse 做物理加载合并；
3. 若多个 query 的 selected blocks 取 union，必须保留 per-query selection mask；
4. QSA 的 causal tail 仍应独立处理，不能因为 workgroup 共享 `kv_stop` 而删除 query-row mask；
5. QSA 的 compressed summary cache 与 XAttention 的 raw KV estimate 是两种不同的选择来源；
6. QSA 的 `selected_blocks` 是 compact metadata，而 XAttention 的 mask 是 PA execution metadata，二者不应混用。

推荐的 CM 设计仍是：

```text
Q0 qsa_kv_cache_update
    → 写入主 PA K/V

Q1 qsa_prepare
    → indexer Q、raw K、pending ring、complete summary

Q2 qsa_score_topk
    → per-query compressed-block score
    → block top-k / partition + finalization

Q3 qsa_sparse_attention
    → selected block union load
    → per-query mask
    → online softmax + PV
```

最终结论是：

> **QSA 的稀疏集合在语义上是 per-query-token 的 compact block selection；XAttention 的稀疏集合在执行上是 query-block/workgroup 级的 mask。二者都可以共享物理 KV 加载，但 QSA 必须通过 per-query selection mask 保持 token 级选择语义，而 XAttention 更依赖 merged workgroup mask、PA tile 和 SLM reuse 来换取吞吐。**
