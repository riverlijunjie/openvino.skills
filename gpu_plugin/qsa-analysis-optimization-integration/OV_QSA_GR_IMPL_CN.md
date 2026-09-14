# QSA / Gated Residual GPU Plugin 实现总结

## 1. 文档范围

本文总结 `/home/ov2022/workspace/qsa/openvino.mx` 中 QSA（Qwen Sparse Attention）与 Gated Residual（GR，也称 Hyper Connection）GPU plugin 的最终实现状态，包括：

- OpenVINO GPU internal op、clDNN primitive、graph node 和 CM implementation；
- QSA/GR pattern transformation 的识别边界；
- QSA 的 Q0–Q3 执行流水线；
- GR READ / WRITE / FINAL_MIX 语义；
- Intel Xe2 上的 `cm_dpas` 优化、运行时 dispatch 和 cache layout；
- 单元测试、远程 Arc B580 验证、spill 和 roofline 结果；
- 当前限制和后续可优化方向。

本文描述的是当前已实现并验证的版本。设计阶段曾考虑过的、但最终未采用的路径会明确标注为“未采用”，避免把实验方案误认为生产实现。

## 2. 背景与目标

Qwen4-Exp / Qwen3.8-Flash-Next 类模型同时包含两种特殊机制：

1. **QSA**：先使用独立 indexer 对压缩后的 KV block 进行打分和 top-k 选择，再让主 attention 只访问 selected token 对应的 KV。
2. **Gated Residual**：把普通 residual `[B,T,D]` 扩展为多个 branch `[B,T,C,D]`，在子层前执行 gated read/mix，在子层后使用 write gate 将结果写回各 branch。

这两条路径无法通过普通 OpenVINO 算子简单拼接获得目标性能，因此实现采用以下边界：

- 新增 GPU internal op 和独立 clDNN primitive；
- QSA 和 GR 的核心计算全部由 CM kernel 完成，包括 tiled GEMM、RMSNorm、RoPE、top-k、softmax、gate 和 residual update；
- 不依赖现有 `paged_attention` 或 `scaled_dot_product_attention` 的 primitive/implementation；
- 不调用 oneDNN GEMM；
- 不满足专用 kernel 约束时，不静默地产生错误结果，而是在 implementation validation 阶段拒绝该实现。

## 3. 生产模型参数对齐

实现和测试按照 `/home/ov2022/workspace/qsa/config.json` 以及 Qwen4-Exp 的 Transformers 模型结构对齐。核心参数如下：

| 参数 | 值 | 含义 |
|---|---:|---|
| `hidden_size` | 2560 | GR read 和 indexer projection 的输入宽度 |
| `num_attention_heads` | 24 | 主 attention query head 数 `Hq` |
| `num_key_value_heads` | 2 | 主 attention KV head 数 `Hkv`，使用 GQA |
| `head_dim` | 256 | 主 attention 的 `Dh/Dv` |
| `indexer_n_heads` | 4 | indexer query head 数 `Hidx` |
| `indexer_kv_heads` | 1 | indexer 使用 MQA |
| `indexer_head_dim` | 128 | indexer head dimension `Di` |
| `indexer_budget` | 2048 | selected token budget |
| `indexer_compress_ratio` | 4 | 每个 QSA compressed block 的 token 数 `r` |
| `indexer_block_topk` | 512 | `budget / compress_ratio` |
| `hc_count` | 4 | GR branch 数 `C` |
| `hc_lowrank` | 320 | GR 低秩维度 |
| `partial_rotary_factor` | 0.25 | 主 attention 的部分 rotary 维度比例 |
| `pa_block_size` | 16 | 当前 QSA 测试和主 cache 使用的 page block 大小 |

`compress_ratio` 是逐层属性。模型中只有 full attention / QSA 层使用 QSA；linear attention 层不创建 QSA primitive。模型配置中的 full attention interval 为 4，但 implementation 不假设所有层都具有相同的 cache 或 QSA 状态。

### 3.1 Qwen RoPE 语义

Qwen4-Exp 使用 HuggingFace `rotate_half` 形式的半分式 RoPE，而不是相邻维度交错配对。对 rotary dimension `R`，第 `j` 个频率配对：

$$
(x_j, x_{j+R/2})
$$

对应旋转为：

$$
\begin{aligned}
y_j &= x_j\cos\theta_j - x_{j+R/2}\sin\theta_j,\\
y_{j+R/2} &= x_{j+R/2}\cos\theta_j + x_j\sin\theta_j.
\end{aligned}
$$

因此，`qsa_common.hpp` 中的 `apply_partial_rope` 使用 `(j, j + rotary_dim / 2)` 配对。测试参考实现也独立手写了相同的 HF 风格公式，不能只通过重新排列 cos/sin 表来修正错误的交错配对。

## 4. 总体执行架构

QSA 的完整路径为：

```text
GR READ/MIX
    │
    ├── mixed [tokens, D]
    │       │
    │       └── Q1: qsa_prepare
    │                    │
    │                    └── indexer Q + raw-K + block summaries
    │                                      │
    │                                      └── Q2: score/top-k
    │                                                   │
    │                                                   └── selected blocks
    │
    ├── 外部主 Q/K/V projection
    │       ├── Q_cur ───────────────────────────────┐
    │       └── K_cur/V_cur ── Q0: KV cache update ──┤
    │                                                 │
    │                         Q3: sparse attention ◄─┘
    │                                  │
    │                                  └── attention head output
    │
    └── 独立 o_proj → GR WRITE/COMBINE
```

Q0 与 Q1 没有数据依赖，可以使用相同的输入 event 并行入队；Q2 等待 Q1；Q3 同时等待 Q0 和 Q2。

依赖关系可以表示为：

```text
Q0 ─────────────────────┐
                        ├──> Q3
Q1 ───> Q2 ──────────────┘
```

QSA op 本身不负责主 Q/K/V projection，也不负责 `o_proj`。主 query 和 output gate 由外部 projection 产生，Q3 直接消费 query layout；Q3 输出 attention head layout，之后再经过独立的 output projection。

## 5. 文件与模块布局

### 5.1 OpenVINO op、primitive 与 graph

```text
src/plugins/intel_gpu/include/intel_gpu/op/qsa.hpp
src/plugins/intel_gpu/include/intel_gpu/op/gated_residual.hpp

src/plugins/intel_gpu/include/intel_gpu/primitives/qsa.hpp
src/plugins/intel_gpu/include/intel_gpu/primitives/gated_residual.hpp

src/plugins/intel_gpu/src/graph/qsa.cpp
src/plugins/intel_gpu/src/graph/gated_residual.cpp
src/plugins/intel_gpu/src/graph/include/qsa_inst.h
src/plugins/intel_gpu/src/graph/include/gated_residual_inst.h

src/plugins/intel_gpu/src/graph/registry/qsa_impls.cpp
src/plugins/intel_gpu/src/graph/registry/gated_residual_impls.cpp
```

`primitives_list.hpp` 注册了 `QSA` 和 `GatedResidual` 两个 internal primitive，graph registry 只为它们注册 CM implementation。当前没有 OpenCL 或 oneDNN 的替代 implementation。

### 5.2 CM implementation 与 kernel

```text
src/plugins/intel_gpu/src/graph/impls/cm/qsa/qsa.hpp
src/plugins/intel_gpu/src/graph/impls/cm/qsa/qsa.cpp
src/plugins/intel_gpu/src/graph/impls/cm/qsa/qsa_kv_cache_update.cm
src/plugins/intel_gpu/src/graph/impls/cm/qsa/qsa_prepare.cm
src/plugins/intel_gpu/src/graph/impls/cm/qsa/qsa_score_partition.cm
src/plugins/intel_gpu/src/graph/impls/cm/qsa/qsa_score_tile_dpas.cm
src/plugins/intel_gpu/src/graph/impls/cm/qsa/qsa_score_topk_fused.cm
src/plugins/intel_gpu/src/graph/impls/cm/qsa/qsa_sparse_attention.cm
src/plugins/intel_gpu/src/graph/impls/cm/qsa/qsa_sparse_attention_dpas.cm
src/plugins/intel_gpu/src/graph/impls/cm/qsa/qsa_topk_finalization.cm
src/plugins/intel_gpu/src/graph/impls/cm/include/qsa_common.hpp
src/plugins/intel_gpu/src/graph/impls/cm/include/qsa_dpas_common.hpp

src/plugins/intel_gpu/src/graph/impls/cm/gated_residual/gated_residual.hpp
src/plugins/intel_gpu/src/graph/impls/cm/gated_residual/gated_residual.cpp
src/plugins/intel_gpu/src/graph/impls/cm/gated_residual/gr_read.cm
src/plugins/intel_gpu/src/graph/impls/cm/gated_residual/gr_write.cm
```

`qsa.cpp` 中的 `QSACmImpl` 管理 QSA 所有 stage，`gated_residual.cpp` 中的 `GatedResidualCMImpl` 根据 mode 选择 `gr_read` 或 `gr_write`。

## 6. QSA internal op 与输入契约

`ov::intel_gpu::op::QSA` 的输入顺序固定为 17 项：

| 索引 | 输入 | 逻辑形状/说明 |
|---:|---|---|
| 0 | `MIXED` | `[tokens, D]`，GR READ/MIX 输出 |
| 1 | `QUERY` | `[tokens, Hq*Dh]`，有 output gate 时为每 head 的 `[query, gate]` 交错布局 |
| 2 | `KEY` | `[tokens, Hkv*Dh]`，当前 token 的主 K |
| 3 | `VALUE` | `[tokens, Hkv*Dv]`，当前 token 的主 V |
| 4 | `PAST_LENS` | `[num_seqs]` |
| 5 | `SUBSEQUENCE_BEGINS` | `[num_seqs+1]` |
| 6 | `BLOCK_INDICES` | page table 使用的 physical block ids |
| 7 | `BLOCK_INDICES_BEGINS` | 每个 sequence 的 page table 起点 |
| 8 | `KEY_CACHE` | 主 f16 K cache |
| 9 | `VALUE_CACHE` | 主 f16 V cache |
| 10 | `INDEXER_RAW_K_CACHE` | indexer raw key cache |
| 11 | `INDEXER_SUMMARY_CACHE` | compressed block summary cache |
| 12 | `INDEXER_QK_WEIGHT` | `[Hidx*Di + Di, D]`，indexer Q/K projection |
| 13 | `INDEXER_Q_NORM_WEIGHT` | indexer Q RMSNorm 权重 |
| 14 | `INDEXER_K_NORM_WEIGHT` | pooled block K RMSNorm 权重 |
| 15 | `ROTARY_COS_SIN` | rotary table，cos/sin 交错存储 |
| 16 | `POSITION_IDS` | absolute position 或 mRoPE rank |

主要 config 属性包括：

- `heads_num`、`kv_heads_num`、`k_head_size`、`v_head_size`；
- `hidden_size`、`indexer_heads`、`indexer_kv_heads`、`indexer_head_dim`；
- `indexer_rotary_dim`、`budget`、`compress_ratio`、`qsa_block_tokens`；
- `pa_block_size`、`block_topk`、main attention scale；
- indexer RMSNorm epsilon、是否存在 output gate、是否 causal。

创建 primitive 前会检查配置的一致性，例如：

- `indexer_kv_heads == 1`；
- `budget % compress_ratio == 0`；
- `block_topk == budget / compress_ratio`；
- `qsa_block_tokens == compress_ratio`；
- `indexer_rotary_dim <= indexer_head_dim`。

当前 CM implementation 还要求所有输入和输出使用 f16/bfyx；位置、长度、page table 使用 i32；主 KV cache 不支持量化 scale/zero-point。

## 7. QSA 四个逻辑阶段

### 7.1 Q0：`qsa_kv_cache_update`

Q0 将外部主 attention projection 产生的 `K_cur/V_cur` 写入分页 f16 KV cache。它遵循 paged KV 的地址契约，但拥有独立的 QSA kernel 和实现，不复用 PA primitive。

每个 work item 对应一个 token 和一个 KV head，典型 dispatch 为：

```text
gws = [1, Hkv, wg_count * 16]
lws = [1, 1, 16]
```

对 sequence `s` 的绝对位置 `p`：

```text
pa_page       = p / PA_BLOCK_SIZE
token_in_page = p % PA_BLOCK_SIZE
block_offset  = block_indices_begins[s] + pa_page
physical_page = block_indices[block_offset]
```

Q0 使用运行时的 key/value pitch 和 offset，因此支持 token-major flat 输入，也支持 fused projection 后的带 padding layout。Q0 与 Q1 独立，不能为了省一次 launch 而强行合并：Q1 的自然工作单元是 indexer token/block tile，Q0 的自然工作单元是 token/KV-head copy，二者 dispatch 形状完全不同。

### 7.2 Q1：`qsa_prepare`

Q1 负责 indexer 路径：

```text
mixed
  → indexer Q/K projection
  → indexer Q RMSNorm + partial rotate_half RoPE
  → raw K 写入 indexer cache
  → 完整 block 的 FP32 pooling
  → block K RMSNorm + block-start RoPE
  → summary 写入
```

Indexer projection 是 CM tiled GEMM，不调用 oneDNN。输出沿最后一维切分为：

- `Q: [tokens, Hidx, Di]`；
- `raw K: [tokens, Di]`。

raw K 写入 cache 时不做 token-level RMSNorm，也不应用 RoPE。只有完整的 `r` token block 才会生成 summary。pooling 使用 FP32 accumulator：

$$
\bar{k}_b = \frac{1}{r}\sum_{j=0}^{r-1} k_{b,j}.
$$

随后对 `\bar{k}_b` 做 RMSNorm，并使用 block 起始位置应用 partial RoPE。

Q1 根据运行时 batch 组织选择工作映射：

- **Prefill**：一个 work-group 负责一个 sequence 的一个 QSA block；
- **Decode**：一个 work-group 负责一个 sequence 的当前 append token，只有 block completion 时才回读并 finalize summary；
- **Mixed**：由 `prepare_map` 同时描述 sequence/QSA-block 工作项，允许多 token 和单 token sequence 共存。

Q1 不写主 KV cache，主 KV 更新只由 Q0 完成。

### 7.3 Q2：score 与 top-k

Indexer score 对 query token `i` 和 compressed block `b` 的定义为：

$$
I_{i,b} = \frac{1}{\sqrt{D_i}}
\sum_{h=1}^{H_{idx}}
\operatorname{ReLU}\left(q_{i,h}^{\mathsf T}\bar{k}_b\right).
$$

必须保持以下顺序：

1. 每个 indexer head 单独计算点积；
2. 对每个 head 应用 ReLU；
3. 沿 indexer head 做 reduce；
4. 最后乘 `1/sqrt(Di)`；
5. 对合法 block 做 top-k。

block causal 条件为：

$$
start_b + r - 1 \le i.
$$

也就是说，只有完整可见的 compressed block 才能参与 block top-k；当前未完成 tail 不参与 block top-k，而是在 Q3 中额外追加。

Q2 有三种实际路径：

#### Prefill 标量路径

`qsa_score_topk_fused.cm` 执行 score 和 top-k 的标量/向量版本，适用于 Xe1 或不满足 dpas 对齐约束的 shape。

#### Prefill dpas 路径

`qsa_score_tile_dpas.cm` 只负责 dpas score 计算，随后统一复用 `qsa_topk_finalization.cm` 完成选择。这样避免在同一个 work-group 内把 global score 写出后又读回；此前的 global-memory round-trip 会在部分驱动上产生不稳定的跨线程可见性问题。

Q2 dpas 的 A tile 为 8 个 block。当一个 page 只有 4 个 summary（生产配置 `pa_block_size=16, r=4`）时，一个 8-row tile 会拆成多个整页 sub-tile，而不是要求 page summary 数量必须整除 8。

block score 行 stride 会向 dpas store 宽度对齐：

```text
padded_stride = ceil(max_blocks / 8) * 8
```

否则最后一个宽 store 可能越过当前行并覆盖下一个 token 的 score。

#### Decode 路径

decode 采用：

```text
qsa_score_partition → qsa_topk_finalization
```

`qsa_score_partition` 沿 block range 划分 partition，默认每个 partition 扫描 512 个 summary；禁止沿 `Hidx` 切分，因为 ReLU 在 head reduction 之前，按 head 切分会重复读取 summary 并改变归约边界。`qsa_topk_finalization` 负责跨 partition 合并并写出 selected block ids 和 selected counts。

### 7.4 Q3：`qsa_sparse_attention`

Q3 直接读取 selected block metadata，并执行：

1. selected QSA block 展开为 token 位置；
2. token 位置映射为 PA page 和 page 内 offset；
3. 间接读取主 f16 K/V cache；
4. 主 attention QK；
5. causal/tail mask；
6. FP32 online softmax；
7. V 加权累加；
8. attention output gate；
9. 写出 `[tokens, Hq*Dv]` attention head output。

无效 selected slot 在地址计算之前被跳过，不能依赖哨兵 page 或越界后再 mask。`o_proj` 不属于 Q3，由外部独立路径执行。

主 attention 是 GQA：

```text
hkv = h / (Hq / Hkv)
```

Q3 使用 `Hkv` 计算 cache 地址，不能沿用 indexer 的单 key head 假设。

## 8. GR 实现

### 8.1 GR 语义

物理输入使用 row-major `[tokens, C*D]`，逻辑上等价于 `[B,T,C,D]`。当前实现首版只接受 `C=4`。

对每个 branch 做独立 grouped RMSNorm：

$$
\bar{R}_c = R_c \cdot
\frac{g_c}{\sqrt{\operatorname{mean}(R_c^2)+\epsilon}}.
$$

READ 的低秩 gate 为：

$$
G = \operatorname{sigmoid}\left(
W_{up}\operatorname{SiLU}\left(W_{down}\bar{R}/C\right)
\right).
$$

mixed 输出为：

$$
X = \frac{1}{C}\sum_{c=0}^{C-1}G_c\odot\bar{R}_c.
$$

WRITE gate 为每 branch 一个 scalar：

$$
S = 2\operatorname{sigmoid}(W_{inject}\bar{R}/C),
$$

最终写回：

$$
R'_c = R_c + S_cY.
$$

三个 `1/C` 都是模型语义，不能省略。read gate 是逐 channel 的 `[C,D]` gate，write gate 是逐 branch scalar，二者不能混为同一种 gate。

### 8.2 GR mode 与 kernel

`GatedResidual` 支持三个 mode：

- `READ`：执行 branch norm、低秩 read gate 和 branch reduce，可选产生 write gate；
- `WRITE`：消费 residual、子层输出 `Y` 和 write gate，执行 branch update；
- `FINAL_MIX`：模型末尾的只读 mixer，复用 READ kernel 但不产生 write gate。

实际 CM kernel 为：

```text
gated_residual/gr_read.cm
gated_residual/gr_write.cm
```

`gr_read.cm` 内部融合：

```text
branch-wise RMSNorm
→ W_down GEMM
→ 1/C
→ SiLU
→ W_up GEMM
→ sigmoid
→ gate * normalized branch
→ branch reduce * 1/C
```

`gr_write.cm` 内部融合：

```text
读取 residual 和 write gate
→ 每个 branch 读取 Y
→ S_c * Y
→ residual add
```

当前 transformation 生成 WRITE primitive 时，write gate 已由上游图计算并作为显式输入传入；因此 WRITE kernel 不重算 `W_inject`。READ/WRITE 不能合并，因为它们之间存在 attention、GDN 或 MLP 子层依赖。

GR implementation 的限制包括：

- `hc_count == 4`；
- `hidden_size` 非零且按 16 对齐；
- `C*D <= 16384`；
- READ/FINAL_MIX 的 lowrank 按 16 对齐且不超过 512；
- 所有输入输出为 f16/bfyx。

## 9. Pattern transformation

实现文件：

```text
src/plugins/intel_gpu/src/plugin/transformations/qsa_gr_fusion.hpp
src/plugins/intel_gpu/src/plugin/transformations/qsa_gr_fusion.cpp
```

### 9.1 GR READ fusion

`GatedResidualReadFusion` 识别以下图结构：

```text
reshape residual
→ grouped RMS
→ reshape
→ W_down MatMul
→ 1/C
→ Swish/SiLU
→ W_up MatMul
→ Sigmoid
→ gate * normalized residual
→ reshape
→ ReduceSum(branch axis)
→ 1/C
```

匹配时验证：

- branch 数为 4；
- branch 和 hidden size 可由静态 reshape 证明；
- reduce axis 确实是 branch axis；
- 两个 `1/C` 都存在且数值正确；
- W_down/W_up 形状为 `[L,C*D]` 与 `[C*D,L]`；
- RMS gain 是展开后的 `[C*D]`；
- MatMul transpose 属性符合 kernel contract。

如果无法证明这些约束，保留原始通用图，不强行替换。

### 9.2 GR WRITE fusion

`GatedResidualWriteFusion` 识别：

```text
reshape write_gate → [T,C,1]
reshape Y         → [T,1,D]
广播相乘
reshape residual  → [T,C,D]
Add
reshape            → [T,C*D]
```

必须验证 gate 在 branch 轴广播、`Y` 在 channel 轴广播，不能接受错误的 broadcast 方向。

### 9.3 QSA detection

`QSAFusion` 识别 indexer score 的特征签名：

```text
MatMul
→ ReLU per indexer head
→ ReduceSum over indexer heads
→ Multiply by 1/sqrt(Di)
→ TopK
```

匹配成功后写入 runtime info：

- `qsa_indexer_detected`；
- `qsa_indexer_head_dim`；
- `qsa_block_topk`。

当前 pass 不会从普通分解图中猜测完整 QSA cache、page table、RoPE table 和主 attention 输入，因此不会在缺少这些契约时直接创建完整 QSA op。生产路径可显式导出 `ov::intel_gpu::op::QSA`；generic transformation 主要承担严格识别和发布元信息的职责。

## 10. Implementation manager 与运行时参数

### 10.1 QSA stage 注册

`QSACmImpl` 注册以下 stage：

```text
kv_cache_update
prepare
score_topk_fused
score_tile_dpas
score_partition
topk_finalization
sparse_attention
sparse_attention_dpas
```

构造时按静态 device capability 添加 dpas stage；执行时按 runtime stage 选择：

- generate：partition + finalization + scalar sparse attention；
- prefill/mixed 且 Q2 满足条件：dpas score + finalization；
- prefill/mixed 且 Q3 满足条件：dpas sparse attention；
- 不满足条件：对应 scalar CM variant。

QSA 不回退到 PA 或 SDPA。

### 10.2 动态 batch 与 stage 判定

`update_rt_params()` 从 instance 的实际 `PAST_LENS` 和 `SUBSEQUENCE_BEGINS` 读取运行时数据，而不是依赖 static-shape node 的 `memory_deps`。它计算：

- `num_seqs`、`num_tokens`；
- `prepare_map`：sequence/QSA-block 工作项；
- `score_tile_map`：不跨 sequence 边界的 16-query tile；
- `max_blocks`；
- decode partition 数量；
- 当前阶段是 PREFILL、GENERATE 还是 MIXED。

`requires_update()` 始终返回 true，因为 page table 和 past length 即使在 shape 不变时也会随迭代变化。

### 10.3 静态 internal buffer sizing

internal buffer 必须根据 static layout 的上界分配，而不能根据单次 iteration 的 runtime `past_lens` 分配。当前 buffer 包括：

- Q1 prepare map；
- indexer Q；
- block scores；
- selected blocks；
- selected counts；
- dpas score tile map。

特别是 score row stride 会使用 dpas store 对齐后的宽度，并且 allocation、runtime scalar、所有相关 kernel 必须使用同一个 stride。

## 11. `cm_dpas` 优化

### 11.1 Xe2 DPAS 几何

最终实现使用：

```text
REG_M = 8
REG_K = 16 fp16 elements
REG_N = 16 on Xe2
```

`qsa_dpas_common.hpp` 提供：

- `transpose_8x8`；
- `transpose_8x16`；
- `transpose_16x16`；
- VNNI tile load；
- A tile load；
- VNNI row load。

2D block load 的单行宽度受硬件限制为最多 64 bytes，即最多 32 个 fp16。曾尝试使用 128-byte row，代码可以编译但会返回错误数据，因此 Q2/Q3 都使用合法的 64-byte row 分块。

CM build 使用：

```text
-cmc -Qxcm_register_file_size=256
```

远程 Arc B580 编译时需要设置 `CM_FE_DIR=/mnt/river`。

### 11.2 Q2 dpas

Q2 dpas 的目标是 indexer summary 与 query 的 GEMM-like score 部分。最终策略：

- 只将 score 计算放入 `qsa_score_tile_dpas.cm`；
- selection 统一复用 `qsa_topk_finalization.cm`；
- A tile 跨 page 时拆为 page-sized subtiles；
- score buffer 行 stride 按 8 对齐；
- 当 Xe1、head dimension、page summary 数量或 SLM 条件不满足时，使用 scalar `qsa_score_topk_fused.cm`。

这同时修复了真实生产配置 `pa_block_size=16, compress_ratio=4` 下原先无法使用 8-row dpas tile 的问题。

### 11.3 Q3 dpas：head-major N tiling

Q3 的难点不是 DPAS 指令本身，而是 selected set 是 per-query 的。最初将 16 个 query token 放入 dpas N 轴，会迫使一个 tile 遍历 16 个 token 的 selected union；在生产 top-k 下 union 约放大 15 倍，导致 dpas 反而慢于 scalar。

最终 N 轴改为 `(head, token)`：

```text
HT heads × TT tokens = 16 columns
```

其中：

- 同一 token 的多个 query heads 共享同一个 selected 集合；
- GQA 中同一个 KV head 的 query heads 共享 K/V；
- `HT` 同时需要整除 16 和 `Hq/Hkv`，避免一个 tile 跨 KV head group；
- 生产 `Hq=24,Hkv=2` 时，`Hq/Hkv=12`，选择 `HT=4`，因此 `TT=4`；
- 测试 `Hq=4,Hkv=2` 时，选择 `HT=2`。

Q3 还使用 head-dimension worker split：生产 `Dh=Dv=256` 时多个 worker 共同处理一个 `(tile, head)`，QK partial result 通过 SLM 汇总；softmax 状态在 worker 中保持一致，PV 和输出各自处理自己的 V slice。

这套方式来自 page attention multi-token kernel 的 worker split 思路，避免单线程持有完整 `16 × 256 × fp32` 输出 accumulator。

### 11.4 dpas capability gate

Q2/Q3 dpas 只在满足条件时启用，包括：

- Xe2 或更新架构；
- head dimension 满足 `REG_K` 对齐；
- 2D load 所需的最小宽度；
- QSA block 与 VNNI step 对齐；
- Q3 的 worker split 能在寄存器预算内放下 query/output/KV tile；
- Q2 page summary 数量能安全拆分为合法 tile。

Decode 不启用 Q3 dpas，因为 decode 主要是 memory-bound，且只有很少 query token，优先采用 scalar CM 路径和规则化 cache 访问。

## 12. 性能、spill 与 roofline 结果

测试机器为 Intel Arc B580（Xe2）。测量使用重复执行并取稳定值，避免首次 kernel compile 和 runtime warm-up 污染结果。

### 12.1 Q3 性能

在 `Dh=Dv=256`、长 prefill 场景：

| 配置 | Q3 dpas | Q3 scalar | 结果 |
|---|---:|---:|---:|
| `Hq=4,Hkv=2` | 约 11.60 ms | 约 17.22 ms | dpas 约 1.48x |
| 生产 `Hq=24,Hkv=2` | 约 13.91 ms | 约 21.59 ms | dpas 约 1.55x |

生产形状下端到端测量约为：

```text
dpas total   ≈ 15.59 ms
scalar total ≈ 23.27 ms
```

约为 1.49x 的端到端改善。决定性优化是 head-major N tiling，而不是继续减少少量地址计算或分支。

### 12.2 spill 检查

通过 IGC shader dump 检查最终 `qsa_sparse_attention_dpas` 汇编：

- 没有 GRF spill/fill；
- 只有少量 flag/predicate spill：
  - `flag load 80`；
  - `flag store 13`。

曾尝试缓存 page lookup、预取地址和消除空 tile 分支，但这些优化没有改善耗时，并引入约 1536 bytes GRF spill，最终全部回退。当前结论是：Q3 主要受数据搬运和 selected union 影响，不能只依据 opcode 数量进行优化。

### 12.3 Roofline 判断

在 B580 上，QSA 的主要阶段具有以下性质：

- Q0 是规整 streaming copy，带宽受限；
- Q1 decode 主要受 projection weight bandwidth 限制，prefill 才更接近 compute-bound；
- Q2 decode 扫描所有 compressed summary，是 decode 的主要带宽成本；
- Q3 decode 只读取 selected KV，仍主要受间接访存和带宽限制；
- Q2/Q3 prefill 在较大的 query tile 下才有机会利用 DPAS 算力。

因此用户提出的策略已落实：prefill 中包含 GEMM 的 score/attention 部分优先使用 `cm_dpas`；decode 不强行使用 dpas，而是优先保证 KV/summary 访存和 occupancy。

### 12.4 token size roofline 快速测试

工作区根目录新增 `qsa_gr_roofline.py`，用于快速比较 `token_size=1024/2048/4096` 时 QSA/GR 在 prefill 和 decode 的理论 roofline。脚本默认使用生产配置（`Hq=24`、`Hkv=2`、`Dh=Dv=256`、`C=4`、`lowrank=320`）和 Arc B580 的可调 FP16 峰值/带宽参数；输出每个阶段的 FLOPs、显存流量、算术强度、compute/bandwidth 两个时间下界以及最终 roofline bound。

默认运行会覆盖三个 token size 和两个阶段，并可用 `--csv` 或 `--json` 保存结果。默认模型的 ridge point 为约 `59.2 FLOP/byte`；sweep 显示 QSA Q3 prefill 和 GR READ/MIX prefill 的算术强度分别约为 `11.9` 和 `230--280 FLOP/byte`，而 decode 的 QSA Q2/Q3 与 GR READ/MIX 分别接近 `1`、`11.9` 和 `1 FLOP/byte`，与“prefill 使用 dpas、decode 优先带宽”的实现策略一致。

### 12.5 远程 B580 实测（`DISABLED_roofline`）

为了让上述模型能与真实 kernel 对照，`qsa_gpu_test.cpp` 和 `gated_residual_gpu_test.cpp` 各新增了一个不校验数值、只测量 GPU 执行时间的 `TEST_P` 套件：

```text
qsa_roofline_perf.measures_cm_kernel_time/*             (DISABLED_roofline)
gated_residual_roofline_perf.measures_cm_kernel_time/*  (DISABLED_roofline)
```

方法（v2，按真实 kernel 粒度测量）：

- 复用 `run_qsa`/`gated_residual` 已有的拓扑构造逻辑，但用生产真实值覆盖 `prod_gqa_dims()` 为缩短 CPU reference 耗时而缩小的参数（`hidden_size=2560`、`budget=2048`、`block_topk=512`，GR 用 `D=640`、`L=320`、`C=4`）；
- **QSA 的 `execute()` 在一次融合 primitive 内部真实派发 5 个独立 CM kernel**（`Q0_kv_cache_update`、`Q1_prepare`、`Q2_score`——prefill-dpas 用 `qsa_score_tile_dpas`、decode 用 `qsa_score_partition`——、`Q2_topk_finalization`、`Q3_sparse_attention`），而 `network::get_executed_primitives()` 只暴露最后一个 kernel（Q3）的 event，无法单独测出每个子阶段。为此新增了一个 debug-only 的 `qsa_stage_events.hpp`（`ov::intel_gpu::cm::qsa_debug::take_stage_events()`），让 `qsa.cpp::QSACmImpl::execute()` 把每个 stage 派发时 `stream.enqueue_kernel(...)` 返回的真实 `event::ptr` 记录到一个 `thread_local` 容器里，测试侧再逐个读取、用 `get_profiling_info()` 里 `profiling_stage::executing` 区间得到 GPU 侧真实执行时间；
- GR 只派发 1 个 kernel（READ 或 WRITE 各一次），沿用原来的单 primitive 计时，无需该 hook；
- 2 次 warm-up + **100 次重复**，对每个 stage 分别取平均值（而不是像 v1 那样只有 QSA 合计耗时的中位数）；
- 每个测量点打印一行 `ROOFLINE_JSON {...}`（新增 `"samples":100` 字段），字段与 `qsa_gr_roofline.py` 的 schema 对齐，可直接管道处理；`qsa_gr_roofline.py` 现在按 5 个 QSA stage + 2 个 GR stage 分别计算 FLOPs/流量/roofline bound，并对每个 (component, phase, tokens) 输出 `roofline-sum` vs `measured-sum` 的效率对比。

> **踩坑记录**：给 `thread_local` 容器长期持有 `event::ptr` 而不显式 `wait()` 会在这台 Arc B580 上挂起 GPU（进程卡在不可杀的 `D` 状态，`wchan=__flush_workqueue`），且与迭代次数或 token size 无关，任何规模都会触发。修复方法是在测试侧的每次迭代里，对拿到的每个 stage event 显式调用一次 `wait()` 之后再读 profiling info，不需要改动 `qsa.cpp` 的派发/依赖逻辑本身。细节见 [/memories/repo/qsa_gpu_hang_incident.md](/memories/repo/qsa_gpu_hang_incident.md)（本地 memory，不在仓库里）。

运行方式（对应 `remote_machine.txt` 中的 Arc B580 机器）：

```bash
cd /mnt/river/qsa/openvino.mx
export CM_FE_DIR=/mnt/river
./bin/intel64/Release/ov_gpu_unit_tests \
    --gtest_also_run_disabled_tests --gtest_filter='*DISABLED_roofline*' \
    2>&1 | tee /tmp/qsa_gr_roofline_run.log

# 本地：
python3 qsa_gr_roofline.py --measured-jsonl /tmp/qsa_gr_roofline_run.log
```

2026-09-09 在远程 Arc B580（`odt-huyuan-openvino-ci-74`，commit `480b16cf67` + 本次未提交改动）上实测，12 个测试点、每点 5（QSA）或 1（GR read）个 kernel、每 kernel 100 次迭代取平均，全部 `PASSED`：

| component | phase   | tokens | stage                  | 实测 ms | roofline bound ms | measured/bound |
|---|---|---:|---|---:|---:|---:|
| QSA | prefill | 1024 | Q0_kv_cache_update      | 0.220 | 0.009 | 23.9x |
| QSA | prefill | 1024 | Q1_prepare              | 5.177 | 0.028 | 183.3x |
| QSA | prefill | 1024 | Q2_score                | 0.051 | 0.005 | 10.8x |
| QSA | prefill | 1024 | Q2_topk_finalization    | 0.340 | 0.005 | 73.9x |
| QSA | prefill | 1024 | Q3_sparse_attention     | 49.123 | 4.765 | 10.3x |
| QSA | prefill | 2048 | Q1_prepare              | 9.825 | 0.056 | 173.9x |
| QSA | prefill | 2048 | Q3_sparse_attention     | 211.603 | 18.948 | 11.2x |
| QSA | prefill | 4096 | Q1_prepare              | 19.275 | 0.113 | 170.6x |
| QSA | prefill | 4096 | Q3_sparse_attention     | 869.292 | 37.896 | 22.9x |
| QSA | decode | 1024 | Q0_kv_cache_update       | 0.004 | ~0 | — |
| QSA | decode | 1024 | Q1_prepare               | 0.213 | 0.007 | 29.6x |
| QSA | decode | 1024 | Q2_score                 | 0.377 | 0.000 | 2547x |
| QSA | decode | 1024 | Q2_topk_finalization     | 0.228 | ~0 | — |
| QSA | decode | 1024 | Q3_sparse_attention      | 1.431 | 0.005 | 307.6x |
| GR | prefill | 1024 | READ/MIX | 22.902 | 0.032 | 724.3x |
| GR | prefill | 2048 | READ/MIX | 45.542 | 0.056 | 806.1x |
| GR | prefill | 4096 | READ/MIX | 90.604 | 0.113 | 801.8x |
| GR | decode | 1024/2048/4096 | READ/MIX | 0.973–0.975 | 0.007 | ~135x |

（完整 30 行 QSA + 6 行 GR 明细见 `qsa_full_sweep_100iter.log` / `python3 qsa_gr_roofline.py --measured-jsonl` 的完整输出；上表只挑了每组的代表性/瓶颈行。）

按 (component, phase, tokens) 汇总 roofline-sum 与 measured-sum 的效率：

| component | phase | tokens | roofline-sum ms | measured-sum ms | efficiency |
|---|---|---:|---:|---:|---:|
| QSA | prefill | 1024 | 4.811 | 54.911 | 0.09 |
| QSA | prefill | 2048 | 19.055 | 222.787 | 0.09 |
| QSA | prefill | 4096 | 38.147 | 892.289 | 0.04 |
| QSA | decode | 1024 | 0.012 | 2.253 | 0.01 |
| QSA | decode | 2048 | 0.017 | 4.035 | 0.00 |
| QSA | decode | 4096 | 0.017 | 4.101 | 0.00 |
| GR | prefill | 1024 | 0.058 | 22.902 | 0.00 |
| GR | prefill | 2048 | 0.108 | 45.542 | 0.00 |
| GR | prefill | 4096 | 0.217 | 90.604 | 0.00 |
| GR | decode | 1024/2048/4096 | 0.007 | 0.973–0.975 | 0.01 |

观察（按真实 kernel 粒度重新得出，比 v1 的整体估计更精确）：

- **QSA prefill 的绝对耗时几乎全部来自 `Q3_sparse_attention`**（占比 89–97%），其次是 `Q1_prepare`（约 8–10%），`Q0/Q2_score/Q2_topk` 合计不到 2%；这与 v1 “Q2 平方项应主导 prefill” 的初步猜测不同——细粒度实测显示真正的瓶颈仍在 Q3，其算术强度约 `11.9 FLOP/byte`（偏 bandwidth-bound），measured/bound 比值稳定在 `10–23x`，说明 Q3 kernel 本身效率尚可、有中等优化空间；
- **`Q1_prepare` 的 measured/bound 比值高达 170–183x，是所有 QSA prefill 子阶段里效率最差的**，其算术强度理论上高达 `339–451 FLOP/byte`（应为 compute-bound、理论上可以跑满 118.8 TFLOP/s），但实测吞吐远低于此，说明该 kernel 的实现（block pooling/summary/RoPE）存在明显的算力利用率问题，是 QSA 侧最值得优先优化的目标；
- **QSA decode 里 `Q2_score`/`Q2_topk_finalization` 的 measured/bound 比值达到 1000x–50000x 量级**——这是因为这两个 stage 在 decode 时的理论 FLOPs/流量极小（个位数 token、单 block 粒度），roofline bound 趋近于 0，此时 kernel 启动开销/occupancy 完全主导实测时间，efficiency 数字本身不再有意义，只能定性说明 decode 阶段是 launch-overhead-bound 而非 compute/bandwidth-bound；
- **GR READ/MIX 在 prefill 阶段的 measured/bound 比值达到 724–806x 且随 token size 几乎线性增长**（22.9→45.5→90.6 ms，约 2x/2x），其理论算术强度约 `230–280 FLOP/byte`（应为 compute-bound），但实测远低于理论峰值，说明这个逐 token 的低秩 GEMV 目前主要受 kernel 启动次数（对每个 token 单独 dispatch？）或 occupancy 限制，而不是算力或带宽，是 GR 侧的主要优化候选；
- 汇总效率（roofline-sum / measured-sum）与 v1 报告的整体估计基本吻合（QSA prefill 0.04–0.09，QSA/GR decode 约 0.00–0.01），说明 v1 的粗粒度结论方向是对的，但只有细粒度数据才能定位到具体是哪个 kernel（Q3 主导总量、Q1 效率最差）在拖累整体表现。



## 13. 测试设计与覆盖

### 13.1 QSA GPU 测试

文件：

```text
src/plugins/intel_gpu/tests/unit/test_cases/qsa_gpu_test.cpp
```

测试包含独立 CPU reference：

- `project`：indexer projection；
- `summary`：block pooling、norm、block-start RoPE；
- `select_blocks`：score、causal、top-k、tail；
- `attention`：selected token 主 attention、online softmax 参考；
- HF 风格 `rotate_half` reference；
- output gate reference。

为避免测试因 top-k 选择退化而失去意义，测试确保 `block_topk << n_complete`。当 FP16 cache round trip 与 ReLU 造成 top-k 边界接近时，比较逻辑使用 selection margin 过滤 near-tie token，而不是把合法的浮点差异错误判为 kernel 错误。

覆盖场景包括：

- 单 sequence prefill；
- 小于一个 QSA block 的输入；
- decode；
- 同一序列 prefill 与 decode 的选择结果一致性（各自单独一次 `execute()` 调用）；
- output gate；
- page block 16、compression ratio 4；
- 生产形状的 `Hq=24,Hkv=2,Dh=256`；
- GQA 测试 `Hq=4,Hkv=2`；
- 长 prefill/decode/chunked-prefill；
- 长度 1563、2048、3657、4096，覆盖小于、等于和大于 2048 的序列；
- Q2 scalar/dpas 结果一致性；
- Q3 scalar/dpas 结果一致性；
- **`mixed_batch_prefill_and_decode_together`**：真正的 continuous batching / `QSAStage::MIXED` 场景——同一次 `network.execute()` 里同时提交一条 chunked-prefill 续写序列（`past_len=40,q_len=37`）和一条 decode 序列（`past_len=63,q_len=1`），两者共享同一层权重但各自独占一段 PA page，通过 `run_qsa_batch()`（新增的多序列 topology builder）驱动。此前所有测试的 `past_lens`/`subsequence_begins` 都硬编码为单序列，`any_multi && any_single` 触发的 MIXED 分支从未在真实 GPU 上跑过，这个测试补上了这个缺口；
- **`dpas_score_multi_chunk_matches_single_chunk`**：通过环境变量 `QSA_TEST_SCORE_BUDGET_BYTES`（仅测试可见，生产路径从不设置）把 Q2 的 `BLOCK_SCORES` scratch 预算临时降到 64 字节，强制一次 64-token prefill 规划出 64 个 query-row chunk 而不是默认 128 MiB 预算下总是恰好 1 个 chunk，从而对 `QSAScoreChunk` 的多 chunk 串行调度和 `force_stage_refresh()` 逻辑提供一个可长期回归的自动化测试（此前这条路径只在开发时临时改小常量手工验证过一次）。

测试支持 `QSA_ITERS`，用于重复执行和性能测量，但它不改变 kernel 语义。

### 13.2 GR GPU 测试

文件：

```text
src/plugins/intel_gpu/tests/unit/test_cases/gated_residual_gpu_test.cpp
```

覆盖：

- branch-wise grouped RMSNorm；
- `1/C` 低秩缩放；
- SiLU、read gate、branch reduce；
- write gate 与 residual update；
- f16 CM 输出与 CPU reference 对比；
- READ/WRITE 的 layout 和 mode 行为。

### 13.3 Transformation 测试

文件：

```text
src/plugins/intel_gpu/tests/unit/transformations/qsa_gr_fusion_test.cpp
```

覆盖：

- GR READ 成功融合；
- 缺少任一 `1/C` 时拒绝融合；
- 非 `C=4` 时拒绝融合；
- GR WRITE 正确 broadcast 时融合；
- 错误 gate broadcast axis 时拒绝融合；
- QSA indexer signature 成功识别；
- scale 位于错误位置时拒绝识别。

## 14. 最终验证记录

在远程 Arc B580 上完成验证：

- QSA/GR 相关测试连续 3 次运行，每次 45 个测试全部通过；
- paged-attention 回归测试 187 个通过；
- QSA 长序列、生产 GQA 和 dpas 路径均完成验证；
- 本地与远程最新实现文件的 md5 已同步；
- Q3 dpas 汇编检查确认没有 GRF spill。

仓库中已知的、与本次 QSA/GR 修改无关的既有失败为：

```text
vlsdpa_gpu_test.mixed_input_runtime_pitches_*
```

该组共 7 个测试在回退 QSA/GR 修改后仍可复现，因此不归因于本实现。

## 15. 编译与 CM 工程注意事项

### 15.1 多 stage CM program 的 include guard

同一个 primitive 的多个 CM stage 会被拼接到一个 compilation unit。`#pragma once` 在这种 codegen 模式下不能可靠地阻止重复定义，公共 CM header 使用经典形式：

```cpp
#ifndef QSA_COMMON_HPP
#define QSA_COMMON_HPP
...
#endif
```

带函数体的宏，例如 `ATTR`，不能放入会被 codegen 自动 `#undef` 的 include guard 内；它们需要在各个 `.cm` 文件中重新定义或放在 guard 外部。

### 15.2 CMake header dependency

CM codegen 依赖必须同时包含 `*.h` 和 `*.hpp`。如果只 glob `*.h`，修改 `qsa_common.hpp` 或 `qsa_dpas_common.hpp` 后可能继续使用旧的 inlined header，造成“源码已修改但 kernel 行为未变”的假象。

### 15.3 多输出 primitive

GR READ 在 `produce_write_gate` 时有两个输出。primitive 必须同时提供：

- `num_outputs`；
- 每个 output 的 data type；
- 每个 output 的 padding/layout；
- implementation manager 的所有输入输出 format。

否则 output 1 可能被默认推断成错误的 f32 layout。

### 15.4 runtime scalar 顺序

CM kernel 的 scalar push 顺序必须严格匹配 kernel signature。QSA 的 query pitch、query offset、head stride 和 gate delta 都是 runtime scalar，不能硬编码。scalar 顺序错误时可能不会触发显式编译错误，但会导致 `num_tokens==0` 或错误地址计算，最终输出全零或错误。

## 16. 当前限制

当前版本明确限制为：

1. QSA 主 KV cache 只支持 f16，不支持 KV quantization scale/zero-point；
2. QSA indexer 为 MQA，即 `indexer_kv_heads == 1`；
3. GR 只支持 `C=4`；
4. QSA/GR CM implementation 要求 f16/bfyx；
5. dpas 优化只覆盖 Xe2 及以上满足对齐和寄存器/SLM 约束的 shape；
6. decode 主要使用 scalar CM attention，重点是 memory bandwidth，不强行套用 prefill dpas tiling；
7. QSA generic transformation 只识别 indexer signature 并写 runtime info，不会从任意分解图猜测完整 cache/state contract；
8. QSA block 与 PA page 在语义上独立，不能把 selected QSA block id 直接当作主 KV page id，Q3 必须先展开 token 再做 page mapping；
9. 当前测试重点是单层 primitive 和 GPU plugin contract，完整模型级 state reset、beam reorder、speculative rollback 仍需在模型集成层继续补充；
10. 当前 Q1 prepare 和 top-k finalization 仍有进一步带宽和 launch 优化空间。

## 17. 后续工作建议

优先级较高的后续方向：

1. 为完整模型集成补充 QSA state 生命周期、reset、beam reorder 和 speculative rollback 测试；
2. 在不改变 QSA selection 语义的前提下优化 Q1 projection/prepare 和 Q2 finalization；
3. 评估 QSA summary cache 的更深层 page reuse 和 L2 locality；
4. 根据更多 Xe2/Xe3 设备实测重新调节 dpas worker、head tile 和 query tile；
5. 评估 BF16 输入以及主 KV cache 量化路径；
6. 对短上下文和高 selected ratio 场景增加 QSA 自有 dense CM variant，并由 selector 选择 sparse/dense；
7. 将本文中的性能数据扩展为统一 benchmark，记录 kernel occupancy、SLM 用量、memory transaction 和 token/s。

## 18. 总结

当前实现把 QSA/GR 作为 GPU plugin 中独立、可验证的 CM-only backend：

- QSA 采用 `Q0 ∥ Q1 → Q2 → Q3` 流水线；
- Q0 独立更新主 KV cache；
- Q1 完成 indexer projection、raw-K、pooling、norm 和 RoPE；
- Q2 完成 score、causal filtering 和 top-k；
- Q3 直接消费 selected metadata，执行 sparse GQA attention；
- GR 以 READ/FINAL_MIX 和 WRITE 两个 kernel family 实现 branch merge 与 residual update；
- pattern transformation 只在语义和 layout 可证明匹配时融合；
- prefill 的 GEMM 部分优先使用 `cm_dpas`，decode 优先优化带宽；
- Q3 的最终性能来自 GQA-aware 的 `(head, token)` head-major tiling，而不是盲目扩大 query-token union；
- 所有关键路径都已在 Arc B580 上完成功能、性能和 spill 验证。

该实现保留了明确的 fallback 和 validation 边界，为后续扩展 BF16、量化 cache、更多 branch 数和完整模型 state 管理提供了清晰基础。
