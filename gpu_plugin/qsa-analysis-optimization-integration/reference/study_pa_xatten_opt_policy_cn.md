# PA / XAttention CM Kernel 优化方法总结（Prefill & Decode）

> 研究对象：`openvino.mx/src/plugins/intel_gpu/src/graph/impls/cm/` 下的
> Paged Attention（PA）与 XAttention 相关 CM（C-for-Metal）kernel。
> 目的：提炼 prefill / decode 两种场景下的优化策略，用于指导 QSA
> （`qsa/qsa_sparse_attention_dpas.cm` 等）稀疏注意力 kernel 的优化。

涉及文件：
- Prefill 主体：`pa_multi_token.cm` + `include/cm_pa_xe2.hpp`（Xe2）/ `include/cm_pa_xe1.hpp`（Xe1）
- Decode 主体：`pa_single_token.cm` + `pa_single_token_finalization.cm`；`pa_small_q.cm` + `pa_small_q_finalization.cm`
- 公共算子：`include/cm_attention_common.hpp`（`ugemm_KQ` / `ugemm_PV0/1` / 转置 / online softmax）
- XAttention：`xattn_find_block.cm`、`xattn_gemm_qk.cm`、`xattn_post_proc.cm`、`include/find_block.hpp`、`include/xattn_subseq_meta.hpp`
- KV cache：`pa_kv_cache_update_ref.cm`、`pa_kv_cache_reorder_ref.cm`

---

## 0. 硬件与总体思路

BMG / Xe2 dGPU（参考 `remote_machine.txt`）：20 Xe、2900MHz、f16 ≈ 118 TOPS，显存带宽 ≈ 456 GB/s。

核心矛盾：
- **Prefill**：query 数量多，QK / PV 是大 GEMM，**算力（systolic dpas）敏感**，同时 KV 读取量大 → 需要把数据搬运和 dpas 计算充分流水化。
- **Decode**：每个 sequence 只有 1（或少量）query，QK / PV 退化成 GEMV，**几乎完全是访存 / 消息数（message rate）瓶颈** → 需要 split-K 拆 KV、宽消息合并、跨 head 复用。

一条贯穿始终的经验：**dpas 指令本身往往不到 1% 的运行时间，瓶颈是访存延迟和 LSC 消息条数**，所以优化重点是「让更多的 load 同时在飞（in-flight）」以及「用更少、更宽的消息搬同样的数据」。

---

## 1. Prefill 优化策略（`cm_pa_xe2.hpp` / `pa_multi_token.cm`）

### 1.1 dpas 系统阵列做 QK 与 PV
- QK：`St[kv][q] = K · Qᵀ`，`cm_dpas<HF, HF, SystolicDepth=8, RepeatCount, float>`。
  - N 轴 = GRF 宽度（Xe2 为 16），固定填满，作为「独立输出」轴（prefill 里是 query tile）。
  - K 轴 = 16 个 fp16（`REG_K = SystolicDepth * VNNI_WIDTH = 8*2`），沿 head_size 分块累加。
- PV：`O = Pᵀ · V`，V 以 VNNI 排布作为 B 操作数。
- 累加器 `rO` 常驻寄存器（float），online softmax 逐 step 更新。

### 1.2 online softmax（`cm_online_softmax_update`）
- 逐 kv_step 维护 `cur_max` / `cur_sum`，返回 `max_comp`（旧累加器的 rescale 系数）。
- 把 scale 与 `log2(e)` 预折进 Q，使 softmax 直接用 `cm_exp`（即 exp2），省一次乘法。
- `-3.4e38f` 作为 mask 哨兵；对 kv 尾部不足一个 tile 的部分显式置 `-inf`。

### 1.3 【关键】`cm_prefetch` 软件流水（latency hiding）
这是最值得迁移的技巧。每个 kv_step 在计算当前 St / PV 之前，**预取下一 step 的 K/V**：
```cpp
uint32_t prefetch_kv_pos = (kv_pos + kv_step) >= kv_stop ? kv_pos : (kv_pos + kv_step);
prefetch_K.set_base_ptr(k_cache_base + prefetch_block_id * blk_stride);
prefetch_K.set_block_y((prefetch_kv_pos + wg_local_id) % CMPA_BLOCK_SZ);
cm_prefetch<CacheHint::Cached, CacheHint::Cached>(prefetch_K.set_block_x(worker_offset));
```
- V 也同样在 PV 之前逐 `v_offset` 预取。
- 作用：把下一 step 的访存延迟与当前 step 的 dpas + softmax 重叠，让多条 load 同时在飞；对访存/消息瓶颈的 kernel 收益最大。
- 尾部 step 用 `prefetch_kv_pos = kv_pos` 复用，避免越界预取。

### 1.4 SLM 多缓冲 + `cm_sbarrier` 生产者/消费者流水（U8 压缩路径）
`OPTIMIZED_SPARSE_PIPELINE`（`SPARSE_BLOCK_SIZE == CMPA_WG_SEQ_LEN`）里：
- KV 先 dequant（U8→f16，减 zp、乘 dscale）后写入 SLM 环形缓冲（`& 3` 4 buffer，或 head 分区时 `% 3` 3 buffer）。
- 用 `cm_sbarrier(1)` / `cm_sbarrier(0)` 做「signal / wait」两段式屏障，形成
  `load2 → read0 → compute0` 的流水，保证「最多领先一个 buffer」不覆盖正在读的数据。
- 半数线程读 K、半数读 V（`wg_local_id < local_size/2`）并行 dequant。

### 1.5 head-size 分区（`enable_head_size_partition`，head_size==256）
- 单线程放不下整个 head 的 `rO`（256×float 超 16KB matrix 上限 / 逼近 256-GRF 寄存器堆）。
- 把 head_size 拆给 `num_worker` 个线程：每个算 **部分 St**，经 SLM 累加成完整 St，再各自做 online softmax（重复算比再加一次 barrier 便宜），PV/epilogue 各自在自己的 V 切片内完成。
- `rO` 进一步拆成 `rO_lo` / `rO_hi` 两半，绕过 CM 单矩阵 16384 字节上限。

### 1.6 访存布局技巧
- **Q 转置加载**：`cm_load<lsc::Transpose>` 直接把行主内存读成 dpas B 操作数（对 fp16 等价于 VNNI 重排），省一次显式转置。
- **V VNNI 加载**：`cm_load<lsc::VNNI>` 让硬件在 load 时完成 `[k/2][n][2]` 重排。
- **2D block load**（`lsc::block_2d_desc`）：由 surface 高度自动裁边，越界行返回 0，减少显式 mask。
- **St→P 转置**：`transpose_St_to_P_half` 用寄存器蝶形转置（8×8 / 16×16），把 `St[kv][q]` 变成 PV 的 A 操作数 `P[q][kv]`。
- KV cache 里出现随机 NaN 时对未用行显式清零（避免污染 dpas）。

### 1.7 块级稀疏跳过（sparse skip）
- `skip_load(kv_pos)` / `skip_compute(kv_pos)`：按 `SPARSE_BLOCK_SIZE` 粒度查 mask，
  整块被 mask 时直接 `continue`（连 load 都省），只推进 `causal_left`。
- `sparse_block_mask`（compute 级）与 `sparse_block_mask_wg`（load 级）两张 mask 分离，
  让「不加载」与「不计算」可以不同粒度。

### 1.8 多子序列合批派发
- host 端把 `(block_start_pos, subsequence_id)` 成对打包进
  `blocked_q_starts_and_subseq_mapping`，一次 launch 混合多条 sequence；
  每个 workgroup 解析自己的 `subsequence_id` 再取 `past_lens` / `block_indices` 等元数据。
- `gws = [1, num_heads, wg_count*wg_size]`，`lws = [1,1,wg_size]`：
  dim1 = head，dim2 = query-block workgroup。

---

## 2. Decode 优化策略（`pa_single_token*.cm` / `pa_small_q*.cm`）

### 2.1 split-K / flash-decoding：沿 KV 长度切分 + finalization 合并
- decode 时每个 query 只有 1 token，为了喂饱 EU，把 **KV 序列切成多个 partition**，
  每个 partition 由不同 workgroup 并行算出「部分」`max / sum / acc`。
- 单独的 `*_finalization.cm` kernel 再把各 partition 的部分结果做数值稳定合并：
  用全局 `max` 重新校正每个 partition 的 `exp(local_max - global_max)`，加权累加 `acc`，
  最后除以合并后的 `sum`。等价于跨 partition 的 online softmax 归并。
- 好处：把「长 KV、单 query」的低并行度问题转成「多 partition 并行 + 一次轻量归并」。

### 2.2 宽消息、减少 message 条数（访存/消息瓶颈）
- decode 侧是 DRAM 带宽 + 消息数瓶颈：一次 128 元素（256B）load 是 **一条** 消息，
  而 8 次 16 元素 load 是 8 条。`qsa_gr_common.hpp::load_hv<N>` 针对
  16/32/64/96/128/256 提供整段宽读特化（对齐 `pa_kv_cache_update_ref.cm` 的 head size 快路径）。
- KV 以「块内连续」为单位整段取，避免高稀疏度下按行分散读浪费带宽。

### 2.3 小 query（`pa_small_q.cm`）用低 RepeatCount 的 dpas
- `RepeatCount` 不必是 8，可为 1/2/4/8；小 query 场景用较小的 M 轴填 dpas，
  N 轴仍填满 GRF 宽度，避免为凑 tile 而做大量无效计算。

### 2.4 GQA / head group 复用 K/V
- 多个 query head 共享同一 KV head（`kv_head = head / (Hq/Hkv)`）。
- 把一组共享 KV 的 head 放到同一 tile/线程，K/V 只读一次供整组使用，
  减少重复访存（prefill 的 head-tile 也用同一思想）。

---

## 3. XAttention 块选择相关（`xattn_*` / `find_block.hpp`）

### 3.1 QK 打分同样走 dpas + VNNI
- `xattn_gemm_qk.cm` 用与 PA 相同的 dpas tile 形状算 Q·Kᵀ 打分，
  VNNI 排布、systolic depth、累加器布局与 `ugemm_KQ` 一致。

### 3.2 块选择 / top-k / 阈值
- `find_block.hpp` 用向量化 reduce 计算每块得分，再做阈值 / top-k 选择；
  避免全排序，用「求最小/最大 + 游标推进」或 radix 思路（QSA 的 `radix_select_topk` 同源）。
- 选择结果按 **升序 block id** 输出，便于后续多路归并（无需再排序、无需 bitmap）。

### 3.3 后处理与元数据
- `xattn_post_proc.cm` / `xattn_subseq_meta.hpp` 负责把选择结果整理成
  下游注意力 kernel 能直接消费的 per-token / per-subseq 元数据（页表、计数、tile map）。

---

## 4. 可迁移到 QSA 的优化清单（优先级排序）

QSA `qsa_sparse_attention_dpas.cm` 已具备：head-tile（HT heads × TT tokens）、GQA 复用、
head-size worker 分区 + SLM 部分 St 归并、VNNI、dpas QK/PV、online softmax、宽 load。

**尚缺 / 可加强**（★ 为「理论优先级」，实测见下方修正）：

| 优先级 | 技巧 | 来源 | 说明 |
|---|---|---|---|
| ✗ 已否决 | `cm_prefetch` 预取下一 union step 的 K/V | §1.3 | **实测在本 kernel 上是 2.3~2.4x 回退**，已回退。原因见 §6。 |
| ★★ | 减少 / 加宽 LSC 消息（合并 load、更大 2D block、跨 head-group 只读一次 K/V） | §1.6/2.2/2.4 | 本 kernel 是「消息发射率」瓶颈，正确方向是**更少更宽的消息**，而非预取。 |
| ★★ | SLM 多缓冲 + `cm_sbarrier` 流水 | §1.4 | 若把 K/V 先入 SLM 再复用，可进一步隐藏延迟；改造较大，需实测。 |
| ★ | 块级 skip_load / skip_compute 双 mask | §1.7 | QSA 的稀疏性已由 top-k 选择保证，价值相对低。 |
| ★（decode）| split-K + finalization 归并 | §2.1 | QSA decode 目前是标量 `qsa_sparse_attention.cm`，长 KV 时可考虑拆 partition。 |
| ★（decode）| 跨 n_rep head 复用 K/V | §2.4 | 标量 decode 现每个 head 重复读同一 K/V，可合并。 |

---

## 5. 本次尝试与结论（已在远程 BMG 实测）

对 **prefill** 的 `qsa_sparse_attention_dpas.cm` 试加了 §1.3 的 `cm_prefetch` 预取：
1. `include/qsa_dpas_common.hpp` 新增 `qsa_dpas::prefetch_wide<ROWS,COLS>()`；
2. kernel 主 union 循环：复制 `cursor` 到副本、重放 TT 路归并求出**下一 step** 的 union 块，
   对其 K/V 发 `prefetch_wide`；真实游标不动 → 不影响选择状态机；
3. 加 `QSA_PREFETCH` 开关做 A/B。

**实测结果（远程 Arc B580，roofline `Q3_sparse_attention` prefill，100 samples min）**：

| tokens | baseline（prefetch 关） | prefetch 开 | 结论 |
|---|---|---|---|
| 1024 | **48.5 ms** | 112–115 ms | 慢 2.3x |
| 2048 | **208.7 ms** | 495 ms | 慢 2.4x |

正确性：全部 27 个 prefill 用例仍 PASS（数值无误），但性能明显回退，**已完整回退**
（`git diff` 干净，回退后复测 49 / 211 ms 与 baseline 一致）。

---

## 6. 关键教训：为什么 prefetch 在这里反而更慢

- 本 kernel 的瓶颈是 **LSC 消息发射率（message-issue-rate）**，不是「延迟且带宽有余」。
- `cm_prefetch` 会**额外发射 LSC 消息**，与真正的 load 抢占同一发射槽；在消息发射率
  已经打满的 kernel 上，等于把 K/V 的消息数翻倍 → 更慢。
- 逐 step 的 lookahead 还多跑了一遍 TT 路归并（额外标量 + `selected_blocks` gather）。
- 对比之下 PA/XAttention 里 prefetch 有效，是因为它们的 dpas 计算占了可观时间、有 compute
  可以和访存重叠；而 QSA Q3 的 dpas 计算 **< 1% 运行时间**，没有可重叠的 compute。
- **正确方向**：沿 §1.6 / §2.2 / §2.4，做「更少、更宽」的消息与跨 head-group 复用，
  而不是增加消息数。任何「借来的」优化都必须先与 **编译期关闭的 baseline** A/B 再采信。
