# DiffusionGemma GPU 输出为空（全 NaN logits）调查记录

> 范围：DiffusionGemma（`diffusiongemma-26B-A4B-it`）在 GPU 上用
> `modeling_diffusion_gemma.exe --device GPU` 生成时输出为空字符串的根因调查。
> 环境：远程 Windows GPU 机器 `Local_Admin@10.239.132.229`
> （openvino.pipeline.mx，genai 在 `thirdparty/openvino.genai`）。
> 日期：2026-06-17 起，**2026-06-18 已解决（§6.17）**。
> 状态：**✅ 已解决并端到端验证（§6.17）**。GPU 默认 build（零 env）产出连贯文本
> （prompt "Why is the sky blue?" → "...a phenomenon called **Rayleigh scattering**..."），
> 全程 `nan=0 inf=0`，`FINAL_EXIT=0`。
>
> **真因（终版，§6.13 坐实）= f16 残差主干动态范围溢出**（约第 7 层越过 65504 → inf → RMSNorm
> `0×inf=NaN` → 全 NaN logits → 空解码）。HF 跑 bf16（指数范围=f32）所以不炸。
>
> **修复（终版，§6.17 + §6.18）= 两件套**：(1) `DG_BACKBONE_F32` 残差高速路标 f32（建模侧，已改为
> DiffusionGemma 始终开启）；(2) **GPU 插件让 `RMSFusion` 跳过这些被标记的高速路 norm**，使其保持
> 分解后的 f32 原语（**这是本次会话的决定性发现** —— 标记融合后的 `RMS` 算子为 f32 **无效**，
> RMS GPU kernel 会无视声明类型把输出下采成 f16）。全局保持 f16，只有薄薄的残差主干 + 喂它的 norm 走 f32。
> （§6.7 的 6 处 MoE-f32 插件改动 **经 §6.18 对照验证不需要、已删**：最终标记策略下 MoE 算子收到的仍是 f16。）
>
> ⚠️ **关键教训（§6.12）**：`DG_DUMP_MAXABS` 的 f32 tap **本身改变了计算**，§6.7–§6.11 的全部
> 测量（`h_attn≤43`、`clamp 修复`、`结构性 miscompile`）都是 tap 把残差边提成 f32 的**伪影**。
> 用插件侧 `DG_ELTWISE_PROBE`（读缓冲、不加图算子）+ 无-tap denoise nan 计数重测后，真因回到了
> 第 21 行最初的判断：纯 f16 残差主干溢出。
> **历史记录（§6.5 INT4-asym、§6.6 ASF、§6.8–§6.11 结构性 miscompile）保留为反面教材**，对应结论均已被推翻。

---

## 0. TL;DR（终版，§6.17 已解决）

- **现象**：`--device GPU` 生成的文本为空；`--device CPU` 正常。
- **真因（§6.13 tapless 坐实）**：GPU 默认 `inference_precision=f16` 下 **残差主干 f16 动态范围溢出**。
  残差流逐层累加（embed×√2816≈53 起步 + 每层 attn/MLP/MoE），约第 6→7 层越过 f16 上限 65504 → `inf`
  → RMSNorm `rsqrt(mean(inf²))=0` → `0×inf=NaN` → matmul 扩散 → 解码器输出**全部 logit NaN**
  （`nan = 256×262144 = 67108864`，`max_logit=-inf`）→ argmax 全 0 → entropy 0 → 提前停 → 空串。
  HF 跑 **bf16**（指数范围=f32）所以永不溢出；GPU 被钉死在 f16（bf16 被拒、全局 f32 撞 MoE f16 孤岛）。
- **为什么花了很久**：`DG_DUMP_MAXABS` 的 f32 诊断 tap **本身把残差边提成了 f32**，部分掩盖了溢出，制造
  出"`h_attn≤43` 却 `out=inf`""结构性 eltwise miscompile""clamp 修复"等一整条**伪影链**（§6.7–§6.11）。
  改用 **tapless** 的插件侧 `DG_ELTWISE_PROBE`（读缓冲、不加算子）后真因才显形（§6.12）。
- **最终修复（§6.17 + §6.18，两件套）**：
  1. **残差高速路标 f32**（建模侧 `keep_f32` + norm `keep_output_f32`；`backbone_f32_enabled()` 始终开）。
  2. **GPU 插件让 `RMSFusion` 跳过这些被标记的 norm**，使其保持分解 f32 原语 —— **关键发现**：标记
     *融合后的* `RMS` 算子无效（RMS kernel 恒输出 f16，§6.16），必须阻止融合。
  全局保持 f16，只有薄薄的残差主干 + 喂它的 norm 走 f32（满足"f16 为全局精度"的硬约束）。
  （~~第 3 条 让 MoERouterFused/3gemm/scatter 接受 f32 layout~~ 经 §6.18 对照验证**不需要、已删**。）
- **验证**：默认 build、零 env、256 tokens → 连贯文本（Rayleigh scattering 多段解释），`nan=0 inf=0`，
  `FINAL_EXIT=0`。删掉 6 处 MoE-f32 改动后 e2e 编译/输出/逐步统计**逐字节相同**（§6.18）。
- **历史警示**：commit `e54c881b`（enable fused_router）的 GPU `MoERouterFused` 原本**只支持 f16**，曾被当成死结的
  核心；但最终最小标记策略下 MoE 收到的仍是 f16（router/experts 预处理从不 keep_f32 + `input.to(f16)`），所以
  这条死结**不成立**、不必改 MoE（§6.18）。INT4-asym（§6.5）、ASF（§6.6）、结构 miscompile（§6.8–§6.11）均已被
  证伪，**不要重走**。

---

## 1. 复现与首个证据

直接跑 e2e（`bin/intel64/Release/run_dg.bat`），sample 自带的逐步诊断
（`modeling_diffusion_gemma.cpp` 里的 `[denoise]` / `[commit]` 行）一击命中：

```
[denoise] step 48  mean_entropy=0.0000  accepted=256/256  nan=67108864  inf=0  max_logit=-inf  argmax[0:16]= 0 0 0 ...
[denoise] step 47  ...                                     nan=67108864  inf=0  max_logit=-inf  argmax[0:16]= 0 0 0 ...
[commit]  canvas 1 argmax[0:16]= 0 0 0 ...  eos_in_canvas=0
[stats]   Generated tokens: 256
--- Generated text ---     (空)
```

- `nan=67108864` = `canvas_length 256 × vocab_size 262144` —— **每一个 logit 都是 NaN**。
- sample 里早有针对性注释（`modeling_diffusion_gemma.cpp` 的 `[denoise]` 诊断块）：
  > 全 NaN 行会被 `compute_entropy`（max=NaN，`p>0` 恒 false → entropy 读成 0）
  > 和 `std::max_element`（全 NaN 行返回 index 0）静默吞掉，伪装成「confident argmax 0」。

这解释了"为什么是空输出"：NaN → argmax 全 0 → entropy 全 0 → 提前停 → 全 0 canvas 在
`skip_special_tokens(true)` 下解码为空串。

---

## 2. 逐层二分：溢出始于第一个 full_attention 层

> ⚠️ **本节结论已被 §6.1/§6.3 的逐层 max-abs dump 修正**。用 `--num-layers` 二分只能看到
> "哪个截断深度开始出 NaN"，会把首溢点误判到 full_attention idx5；实测首溢点其实是
> **decoder 残差主干在 l7 越界**（与 full/sliding 无关，是深度累积），且 idx5/idx6 看似"开始变坏"
> 是因为截断模型在该深度尾部恰好越界。保留本节仅作"二分法的局限"反面教材。

用 sample 的 `--num-layers N` 截断 decoder（编译快、显存小，避免 26B 换页），
GPU 上逐 N 测试 `max_logit` / `nan`：

| `--num-layers` | 层类型覆盖 | 结果 |
|---|---|---|
| 2 | 全 sliding | 干净，`max_logit≈37.5/45/56`（=softcap30÷温度），`nan=0` |
| 5 | 全 sliding（idx 0-4） | 干净，`max_logit=37.5`，`nan=0` |
| **6** | **加入 idx 5 = 第一个 full_attention** | **`nan=1310720`(=5×vocab) 开始出现** |
| 8 / 16 / 30 | 含多个 full_attention | 全 NaN（67108864） |

`config.json` 的 `layer_types`：idx 0-4 是 `sliding_attention`，**idx 5 是第一个
`full_attention`**。结论：**NaN 起点 = 第一个 full_attention 层**。

full vs sliding 的关键差异（`processing_diffusion_gemma.cpp`
`resolved_head_dim` / `kv_heads`）：

| 参数 | sliding | full_attention |
|---|---|---|
| `head_dim` | 256 | **512**（`global_head_dim`）|
| kv heads | 8 | 2（`num_global_key_value_heads`）|
| `scaling_` | 1.0 | 1.0（无 1/√d，靠 q_norm/k_norm 控幅） |
| rope_theta | 10000 | 1000000，partial_rotary 0.25 |

head_dim 翻倍（256→512）使 RMSNorm 的 `mean(x²)` 在更大维度上累加，更易越过 f16 上限。

---

## 3. 关键对照实验：定位到 f16 精度而非逻辑 bug

| 实验 | 配置 | 结果 | 结论 |
|---|---|---|---|
| dummy 权重 + CPU | 小幅值(±0.02)，f32 | ✅ 正常 | — |
| dummy 权重 + GPU | 同上含 full_attention 层 | ✅ 正常（`nan=0`，argmax 前两步还与 CPU 一致） | **融合 INT4 MoE / full_attention 逻辑本身正确** |
| 真实权重 + GPU + 2 层 | f16 | ✅ 正常 | 浅层不溢出 |
| 真实权重 + GPU + ≥6 层 | f16 | ❌ 全 NaN | **真实权重幅值 + f16 精度问题** |

dummy（幅值极小）在 GPU 上即使含 full_attention 也不溢出 → 排除算子逻辑 bug，
锁定为**真实权重幅值下的 f16 动态范围溢出**。

---

## 4. 为什么"提精度"类修复都行不通（逐一验证）

| 尝试 | 结果 |
|---|---|
| `ov::hint::inference_precision = f32` | **编译失败**：`[GPU] No layout format available for moerouterfused:MoERouterFused_xxxx ... (data_type: f32)` |
| `ov::hint::execution_mode = ACCURACY`（→ inference_precision=dynamic） | **同样编译失败**，dynamic 仍把 f32 传到 `MoERouterFused` |
| `ov::hint::inference_precision = bf16` | GPU 直接拒绝：`Supported values: { f16, f32, dynamic }` |
| `ov::hint::activations_scale_factor = 8.0` | 能编译，但**没修好**；8.0 时反而让 step3 也变全 NaN。**后续 §6.6 扫描 8/16/32/64 全部失败**且把 encoder 也弄溢出 → ASF 出局 |

**核心约束**：commit `e54c881b`（enable fused_router for diffusion_gemma）启用的 GPU
`MoERouterFused` 算子**只支持 f16**
（`moe_router_fused_opt.cpp`：`MOE_DTYPE = data_type==f16 ? "half" : "float"`，
但 program_builder 给不出 f32 layout）。这使得"全局切 f32"这条最直接的路被堵死。

> 注（已更正）：早先以为 sample 的直接路径绕过了 `activations_scale_factor`。复查后确认
> `create_diffusion_gemma_encoder_model` / `..._decoder_model` 末尾都调了
> `apply_runtime_options(model, f16, 8.0f)`（`runtime_options.cpp:54`），
> **已经给图设了 `activations_scale_factor=8.0` 和 `kv_cache_precision=f16` 的 rt_info**。
> 也就是说 6.1 的逐层 dump 是在**带 ASF=8.0** 的图上测得仍然溢出 ——
> 单靠 ASF 不足以救深层手写 RMSNorm，必须叠加精度敏感标记（见 §7.1）。

---

## 5. 根因机理：手写 RMSNorm 在 GPU f16 下被降精度

对比两类 RMSNorm 实现：

- **规范实现**（`layers/rms_norm.cpp`，被 qwen3_next / qwen3_5 使用）：发射融合的
  `ov::op::internal::RMS` 算子 —— GPU 内部以 f32 累加平方和，**无论 I/O 是 f16 都不溢出**。
- **DiffusionGemma**（`modeling_diffusion_gemma_text.cpp` 的 `DiffusionGemmaRMSNorm::forward`
  与 `detail::rms_norm_no_scale`）：**手写分解原语** `pow(2) → mean → rsqrt → mul`。
  虽然代码里 `.to(f32)` 上提，但在 `inference_precision=f16` 下，GPU 插件的
  **fp16-compression（ConvertPrecision）pass 会把这些 f32 孤岛重新降回 f16**，于是
  `x²`（在 hidden=2816 或 head_dim=512 上 reduce）越过 65504 → `inf` →
  `rsqrt(inf)=0` → `0×inf=NaN`。

旁证：RoPE 的 cos/sin 已经通过 `ops::precision::disable_fp16_compression_subgraph`
保护（`precision.cpp:50`），而这些 RMSNorm 没有保护。

> 不能简单把 DiffusionGemma 的 norm 换成融合 `ov::op::internal::RMS`：实测会**破坏 CPU 编译**
> —— `DecomposeRMSNorm` 在无 gamma 的 4D `v_norm`（`f32[?,8,?,256]`）上抛异常。
> DiffusionGemma 是**有意**手写 norm 的（v_norm 是无 scale 的 RMSNorm 作用在 4D V 上）。

---

## 6. 历史：早先的部分修复（有效但不彻底）

最初只给**两处**手写 RMSNorm（`rms_norm_no_scale` + 一处 `DiffusionGemmaRMSNorm::forward`）
的方差子图打精度敏感标记：

```cpp
auto sq  = xf.pow(2.0f);
auto var = sq.mean(-1, true);
auto inv = (var + eps).rsqrt();
ops::precision::disable_fp16_compression(sq);   // x^2 才是真正的溢出点
ops::precision::disable_fp16_compression(var);
ops::precision::disable_fp16_compression(inv);
```

> 注意必须包含 `sq`（`pow(2)` 的 Power 节点）—— 起初只标 `var`/`inv` 无效，因为溢出发生在
> `sq` 这个未命名中间节点上。也**不能**用 `disable_fp16_compression_subgraph(inv)`：
> 它会沿 `xf` 一路回溯整张上游网络，把所有层都标成 f32，重新触发 MoE 的 f32-layout 报错。

**当时效果**：NaN 从全部 256 降到 step3 约 5 个，但 step2/1 仍全 NaN。
**当时误判**：以为还有"未覆盖的溢出点"（MLP/SDPA matmul）。6.1 的逐层 dump 证明并非如此 ——
真正漏掉的是 (a) **深层** post-FFN norm（覆盖不全），和 (b) **self-conditioning 反馈放大**
（host 端 `compute_soft_embeddings` 对 NaN 不设防）。完整修复见 §6.2。

---

## 6.1 逐层 max-abs dump：精确定位（2026-06-17 二次调查）

给 encoder/decoder 每层 forward 插入 `max(|hidden|)` 标量输出（env `DG_DUMP_MAXABS` 开启，
`=full` 再加注意力内部细粒度 tap）。tap 子图强制 f32 + `disable_fp16_compression`，并把
**NaN 折叠成 +inf**（GPU `ReduceMax` 会忽略 NaN，必须 `where(x!=x, +inf, |x|)` 才能看见），
这样越界点会以 `inf` 暴露而不是被 f16 上限 clamp 掉。实现见
`modeling_diffusion_gemma_text.cpp` 的 `detail::tap_maxabs` / `maxabs_scalar`，打印见 sample 的
`print_dbg_taps`。

**实测结论（`--num-layers 8 --max-denoising-steps 3`，GPU，真实权重，无精度保护 baseline）：**

| 阶段 | 首个非有限 tap | 说明 |
|---|---|---|
| **encoder** | 无（max≈1.9e3 @`l1.h1`） | **encoder 根本不溢出**（推翻了"首溢在 encoder full_attn idx5"的旧假设） |
| **decoder step3**（self-cond=0） | `l7.out_prescale`(inf) | l0–l6 全有限，**只有最深的 l7 溢出**；只坏 7/256 个 logit（`nan=1835008=7×vocab`），`max_logit=37.5` 仍健康 |
| **decoder step2**（self-cond 激活） | `l0.self_cond`(NaN) | `embed`=16 有限，**NaN 从 self-conditioning 入口就出现** → 扩散到全部 256 → 全 NaN |
| **decoder step1** | `self_cond`(NaN) | 同上 |

**l7 内部细粒度（step3，`=full`）**：`attn_out=53 → h_attn=43 → mlp_out=12 →`
**`h1=2100`** `→ moe_out=2.9 →` **`h2=634`** `→ h_ffn=30 →` **`out_prescale=inf`**。

两个关键机理：

1. **首溢点 = 深层 post-FFN norm/残差尾部**。`out_prescale = h_attn + h_ffn`，而 `h_attn=43`、
   `h_ffn=30` 的逐元素和不可能是 inf。tap（额外 f32 消费者）在被测节点处**阻止了溢出**、但原生
   f16 路径仍溢 —— 这恰好证明溢出就在 l7 的 post-FFN norm 链里。
   proximate 大幅值是 **`h1≈2100`（l6 甚至 2700）= `post_feedforward_layernorm_1(mlp_out)`**：
   这是个**大 gamma（≈175×）的手写 RMSNorm**。残差流带着这些 ≥256 的值进入下一个未保护的手写
   RMSNorm，`x²` 立刻越过 65504。**逐深度门控**：残差流随层数累积，只有最深的 l7 先越界。
2. **encoder 不溢、decoder 溢 = 输入幅值差异**。decoder 跑在**随机噪声 canvas**（random token
   ids 初始化）上，激活比 encoder 处理的连贯 prompt 更大更乱 → 只有 decoder 越界。这解释了
   "为什么 CPU/encoder 看着没事、唯独 GPU decoder 空输出"。

**self-conditioning 反馈是放大器（致命的一环）**：step3 那 7 个 NaN logit 经 sample 的
`compute_soft_embeddings`（NaN 行的 softmax→NaN soft-embed）反馈成 `self_cond_embeds`；step2
一进 `DiffusionGemmaSelfConditioning::forward` 就 NaN（`l0.self_cond` 即首溢），瞬间扩散到全部
256 个位置 → 全 NaN → 空串。**即使 step3 只坏 7 个位置，反馈环也会在 1~2 步内放大成全坏。**

---

## 6.2 完整修复（已验证有效，2026-06-17）

按 6.1 的两个机理双管齐下：

**修复 A —— 保护所有手写 RMSNorm 的方差子图**
（`modeling_diffusion_gemma_text.cpp`）。把 6 节的标记提成 helper `detail::protect_rms_variance(sq,var,inv)`，
并在 **两个** norm 入口都调用：

- `detail::rms_norm_no_scale`（v_norm + router 用）
- `DiffusionGemmaRMSNorm::forward`（input / post_attention / 各 pre/post_feedforward norm 用 ——
  **深层这些才是漏点**）

```cpp
inline void protect_rms_variance(const Tensor& sq, const Tensor& var, const Tensor& inv) {
    ops::precision::disable_fp16_compression(sq);   // pow(2) 是真正越界点
    ops::precision::disable_fp16_compression(var);
    ops::precision::disable_fp16_compression(inv);
}
```

**修复 B —— host 端 self-conditioning 反馈做 NaN-safe**
（`modeling_diffusion_gemma.cpp::compute_soft_embeddings`）。某个 logit 行含非有限值时，
该位置 soft-embed 直接置 0（等价于"该位置不做 self-conditioning"，与 step 首步同语义），
**把单点越界与全 canvas 崩溃解耦**：

```cpp
bool row_finite = true;
for (size_t i = 0; i < vocab_size; ++i)
    if (!std::isfinite(row[i])) { row_finite = false; break; }
if (!row_finite) {  // 不要把 NaN 喂回下一步 decoder
    std::memset(output.data() + pos * hidden_size, 0, hidden_size * sizeof(float));
    continue;
}
```

**验证结果**（`--num-layers 8 --max-denoising-steps 3`，GPU，真实权重，dump 开）：

| 步 | 修复前 | 修复后 |
|---|---|---|
| step3 | `l7.out` inf，7/256 NaN | 仍有 7 个瞬态 NaN（截断到 8 层时深层 norm 仍偏大），但 `max_logit=37.5` 健康 |
| step2 | **全 256 NaN** | **`no-overflow`，nan=0**，`max_logit=45`，argmax 多样（5/173/48…） |
| step1 | **全 256 NaN** | **`no-overflow`，nan=0**，`max_logit=56`，argmax 多样 |
| 输出 | 空串 | **非空**（8 层截断下是乱码，但证明 NaN 级联已断） |

→ 修复 B 在 8 层截断下断链：step3 瞬态 NaN 不再经反馈放大。**但这只对截断模型成立**（见 §6.3）。

---

## 6.3 全 30 层 e2e 反例：修复在真实深度下不够（2026-06-17）

把同样两步修复跑**全 30 层**（不截断，`--output-tokens 64`，默认 48 denoising steps，dump 关）：

```
[denoise] step 48  mean_entropy=0.0000  accepted=256/256  nan=67108864  inf=0  max_logit=-inf  argmax= 0 0 0 ...
[commit]  canvas 1 argmax= 0 0 ...   --- Generated text ---   (仍为空)
```

**关键观察**：step 48 是**第一帧、self-cond=zeros**（修复 B 的反馈防护此时根本不参与），却已经
**全 256 NaN**。这与 §6.1/§6.2 的 8 层结论矛盾，说明：

- **8 层截断严重低估了问题**。`--num-layers 8` 把 26B 压成浅模型，残差流没积累到全深度的量级，
  所以截断下"只有 l7 尾部偶发 7 个 NaN"，修复后看似断链；真实 30 层下溢出更早、更广。
- step48 无 self-cond 仍全 NaN ⇒ 这是**纯前向溢出**，且 **§6.2-A 的"norm 内部 pow(2) 提 f32"
  没拦住**。最可能：**残差 hidden 流本身在深层就已 f16 溢出成 `inf`**——一旦 norm 的*输入*是
  `inf`，把 norm 内部算成 f32 也无济于事（`inf` 进 `inf` 出）。换言之要保护的不是 norm 内部，
  而是**残差主干**（embed_scale·√hidden=√2816≈53 起步，逐层 + attn/MLP/MoE 累加）。

**教训**：截断实验只能定位"截断模型"的首溢点，**不能外推到全模型**；深度累积型溢出必须在全深度复现。

### 全深度 dump 结果（30 层，`--max-denoising-steps 1`，dump 开）

```
[maxabs encoder]        no-overflow  max_finite=1.942e+03 @l1.h1
[maxabs decoder step 1] FIRST_NONFINITE=l7.out_prescale  max_finite=2.696e+03 @l6.h1
[denoise] step 1  nan=67108864(全部)  max_logit=-inf   --- Generated text ---  (空)
```

**关键结论（修正 §6.1/§6.2）**：

1. **首溢点在全深度下仍是 `l7.out_prescale`**（残差主干 `out = h_attn + h_ffn`），与 8 层一致；
   但全深度下 l7 之后还有 22 层注意力，把 l7 的少量 NaN **横向扩散到全部 256** → 全 NaN → 空。
   8 层时 l7 是末层、NaN 不扩散，才只坏 7 个、显得"修好了"。
2. **§6.2-A（norm 内部 pow(2) 提 f32）没拦住 l7 溢出**。溢出不在 norm 内部，而在
   **残差主干本身**：`out_prescale = h_attn + h_ffn`，逐元素和按 tap 读数应 ~40，却是 inf ——
   说明真正越界的是主干上累积的 f16 张量，**norm 的输入早已偏大，把 norm 内部算成 f32 也救不了**。
3. **根因升级**：这是 **Gemma 系特有的大残差流** 问题 —— embedding 先乘 `embed_scale=√hidden=√2816≈53`，
   再逐层叠加 attn/MLP/MoE，残差幅值随深度单调增长，**约在第 7 层冲破 f16 上限 65504**。
   HF 用 **bf16**（指数范围同 f32，max≈3e38）所以无事；GPU 这里被钉死在 f16
   （bf16 被拒、f32 触发 MoERouterFused layout 报错），残差主干无处容身。

→ 因此 **§6.2 的修复方向（保护 norm + 反馈防护）治标不治本**。真正要保护的是**残差主干精度**，
但主干又直接喂给 f16-only 的 `MoERouterFused`（router_input = 原始 `h_attn`），
**一旦把主干标 f32 就重新触发 MoE layout 报错** —— 这正是本问题的死结。

---

## 6.4 验证死结：禁用 fused MoE 后 f32 能编译（但 eager f32 MoE 撑爆显存）

给 sample 加了两个 env 开关（只改 sample，可逆）：

- `DG_DISABLE_FUSED_MOE=1`：GPU 上也强制走 **eager dense MoE**（去掉 f16-only 的 `MoERouterFused`）。
- `DG_INFER_PRECISION={f32|f16|dynamic}` / `DG_ACCURACY=1`：设 `inference_precision` / `execution_mode`。

实验（`DG_DISABLE_FUSED_MOE=1 DG_INFER_PRECISION=f32`，10 层）：

```
[diffusion_gemma] DG_DISABLE_FUSED_MOE set: forcing eager dense MoE
[diffusion_gemma] DG_INFER_PRECISION=f32
[compile] encoder GPU in 236601 ms          <- f32 编码器编译成功，无 layout 报错！
Error: [GPU] ProgramBuilder build failed!    <- 解码器编译 OOM
[CL ext] Can not allocate 2030043136 bytes for USM Host
```

**两点决定性结论**：

1. **去掉 fused MoE 后，f32 确实能编译**（encoder 成功，无 MoERouterFused layout 错）——
   坐实了"f16-only 的 `MoERouterFused` 是 f32 的唯一拦路虎"这一死结诊断。
2. **但 eager dense f32 MoE 显存爆**：2GB 单次分配失败。eager 路径把 experts 平铺成
   `[E=64, T=256, H=2816]` f32 ≈ 1.8GB/层，加上同时持有 f32 encoder 的常量，host RAM 超顶。
   （eager 路径本就是给 CPU 兜底用的，不为 GPU 大 canvas 设计。）

→ 所以"全局 f32"在本机不可行（显存），但**思路被证明是对的**。真正可落地的修复见 §7。

---

## 6.5 排除 INT4-asym 量化为溢出根因（对照实验，2026-06-17）

> 假设：`l7.out_prescale` 的越界可能是 **权重 INT4-asym 量化** 注入的误差/异常值放大造成，
> 而非残差主干固有的大激活。做**同深度同输入**对照（仅改 MoE 精度）证伪。

**先厘清量化在本模型里的真实作用域**（读 `safetensors_weight_finalizer.cpp` /
`quantization_selector.cpp` / `modeling_diffusion_gemma_text.cpp`）：

- 盘上权重是 **FP8（`f8e4m3`）**，量化是**在线（in-flight）**的。
- attention/MLP 的 INT4 由 env `OV_GENAI_INFLIGHT_QUANT_MODE` 驱动；**MoE experts 的 INT4-asym
  与 env 无关**，只要 `OpPolicy::use_fused_moe`（GPU）就量化（`group_size_=pick_moe_group_size`）。
- ⚠️ **`run_dg.bat` 的 env 量化其实没生效**：cmd 的 `set OV_GENAI_INFLIGHT_QUANT_MODE="int4_asym"`
  会把**引号一起**存进变量（`getenv` 读到 `"int4_asym"`），而 `parse_quantization_config_from_env`
  只 `toupper` 后直接比较 `=="INT4_ASYM"`（**不去引号/空格**）→ 落到 `Mode::NONE`。两次运行的日志都打
  `Quantized weights: 0  Quantization coverage: 0.0%` 坐实了这点。**所以 q/k/v/o_proj、MLP 全程
  FP8→F32，从未走 INT4**；唯一真正生效的 INT4 只剩 GPU fused MoE experts 这一条。

**对照实验**（`--num-layers 8 --max-denoising-steps 1`，GPU，真实权重，`DG_DUMP_MAXABS=1`，
仅 `DG_DISABLE_FUSED_MOE` 不同）：

| | Exp A（对照：fused **INT4-asym** MoE） | Exp B（**eager F32** MoE，去掉唯一 INT4 路径） |
|---|---|---|
| MoE experts 精度 | INT4-asym（fused，`group_size=64`） | **F32**（eager dense，`group_size_=0`） |
| 非 MoE 权重 | FP8→F32（env 量化未生效） | FP8→F32（同） |
| FIRST_NONFINITE | `l7.out_prescale` | **`l7.out_prescale`（同位）** |
| max_finite | 2.696e3 @l6.h1 | 2.702e3 @l1.h1 |
| nan | 1835008（7/256） | **1835008（7/256，逐位一致）** |
| max_logit | 37.5 | 37.5 |

**决定性结论**：把**唯一**生效的 INT4 路径整个换成最高精度的 F32 MoE 后，`l7.out_prescale`
**仍在完全相同的位置、以完全相同的数量（7/256）溢出**。

1. **INT4-asym 被证伪**，不是溢出根因。`moe_out≈2.9`，且经 `post_feedforward_layernorm_2`/
   `post_feedforward_layernorm` 两层 RMSNorm **尺度归一**后对 `out_prescale` 贡献被抹平，换 F32 无差别。
2. 机理判据也对不上：INT4-asym RTN（`quantize_moe_int4_asym_view`）是标准 per-group min/max、
   zp∈[0,15]、round-trip 无系统偏置，只注入**有界零均值噪声**（matmul 后误差几个百分点），
   **给不出 `l6→l7` 单层 ~2700→65504 的 24× 突跳**。
3. → 溢出是 **f16 残差主干承载 Gemma 系固有大激活（massive activations）** 的结果（`h1` 由
   `post_feedforward_layernorm_1` 大 gamma≈175× 驱动），**与量化无关**，与 §6.3 的 bf16/f16 论断一致。
   这也修正了"残差平滑累积越界"的措辞——是**某层大激活通道一次性越界**，非平滑爬升。

→ 因此 §7 的修复方向不变（残差主干需 f32 承载 / 让 fused MoE 接受 f32），**不要再往量化方向找**。
   附带可清理项：把 `run_dg.bat` 里 env 量化的引号去掉（`set OV_..._MODE=int4_asym`），或让
   `parse_quantization_config_from_env` 去引号——但这只影响 attention/MLP 是否 INT4，**与空输出 bug 无关**
   （本 bug 在 attention/MLP=FP8→F32 时已复现）。

---

## 6.6 排除 ACTIVATIONS_SCALE_FACTOR（ASF）为可行修复（对照扫描，2026-06-17）

> 假设：OpenVINO 的 `ov::hint::activations_scale_factor`（ASF）能在 f16 推理下把激活整体
> down-scale，避开 `l7` 越界。**结论：在本模型上 ASF 任何取值都修不好，反而把原本干净的 encoder
> 也搞溢出。** 这与根因（gamma 驱动的大激活、尺度不变）一致。

**ASF 机理**（`transformations/common_optimizations/activations_scaling.hpp` + GPU
`transformations_pipeline.cpp:1411-1459`）：`ScaleDownSingleLayer` 把 **MatMul/Conv** 包成
`Mul(1/S) → MatMul → Mul(S)`；`EliminateScalarMul` 利用 `Norm(x·c)=Norm(x)` **只在 matmul 直接喂
Norm 时**丢掉那个 `×S` 上提。**当 matmul 输出流向残差 add（正是我们的溢出点 `out_prescale`）时，
`×S` 上提被保留** → 残差主干恢复满幅值，ASF 帮不到它，反而那个上提 Multiply 自身可能越界。

**两个关键发现（先于实验，靠读码）：**
1. 模型其实**已经**通过 `apply_runtime_options(model, f16, 8.0f)`（`runtime_options.cpp:59`，
   `modeling_diffusion_gemma_text.cpp:1724/1823`）把 ASF=8.0 写进 rt_info。
2. **但 GPU 插件在 LLM 上会丢掉 rt_info 的 ASF**：`execution_config.cpp:170` 的
   `if (!is_llm || (has_lora && !info.supports_immad))` —— decoder 含 KV-cache
   `ReadValue`/`Assign`（`is_large_language_model` 命中）→ 在独显（`supports_immad`）上 **rt_info 的
   8.0 从未下发到 decoder**。所以"早先 ASF=8 没用"的结论不可信：8.0 可能压根没生效。
   → 为此给 sample 加 `DG_ASF=<float>`，**以 user compile property 形式**设 ASF（绕过 is_llm 门、
   且可扫更大值）。验证日志确认生效：`DG_ASF set: activations_scale_factor=...`。

**对照扫描**（`--num-layers 30 --max-denoising-steps 1`，GPU，真实权重，`DG_DUMP_MAXABS=1`）：

| ASF | encoder 首溢 | decoder step1 首溢 | decoder max_finite | nan | 输出 |
|---|---|---|---|---|---|
| **无**（§6.3 baseline） | **不溢**（max 1942） | `l7.out_prescale` | 2.696e3 | 全部 67108864 | 空 |
| 8 | **溢 `l6.out_prescale`** | `l7.attn_out` | 2.692e3 | 全部 | 空 |
| 16 | 溢 `l6.out_prescale` | `l7.attn_out` | 2.796e3 | 全部 | 空 |
| 32 | 溢 `l6.out_prescale` | `l7.attn_out` | 2.690e3 | 全部 | 空 |
| 64 | 溢 `l6.out_prescale` | `l7.attn_out` | 2.742e3 | 全部 | 空 |

**决定性结论：**
1. **任何 ASF 取值都修不好**（8/16/32/64 全是全 NaN→空串）。
2. **`decoder max_finite` 对 S 完全不敏感**（2.69–2.80e3，与无 ASF 基线一模一样）。若 ASF 真在缩放
   残差主干，这个数应随 S 降 8/16/32/64 倍——它纹丝不动 → 证明 `×S` 上提在溢出点**之前**就把幅值
   还原了，主干根本没沾到光。这正是"大激活是 **gamma 驱动的 RMSNorm 输出、尺度不变**"的直接证据。
3. **ASF 反而把 encoder 弄溢出**。无 ASF 时 encoder 从不溢出（max 1942）；ASF 生效后 encoder 在
   `l6.out_prescale` 越界（插入在残差 add 边界的 `×S` 上提 Multiply 自身冲破 65504）；decoder 首溢
   点甚至**提前**（`out_prescale`→`attn_out`）。

→ **ASF 出局**，不是可行修复。本质原因：ASF 只能缩"喂给 Norm 的 matmul"，缩不动残差主干上
   **尺度不变的大激活**；这与 §7 第 3 条"缩放残差主干"的局限是同一回事（gamma 驱动项压不下来）。
   工具：sample 的 `DG_ASF=<float>` 已并入（可逆、默认不设；见 §8）。

---

## 6.7 ✅ 突破：MoERouterFused 支持 f32（§7-1）+ 残差主干标 f32（§7-2）组合，全 30 层 NaN 从 256→9（2026-06-17）

把 §7 的第 1 条和第 2 条**合起来**做，第一次让 GPU f16 推理下残差主干不再全面溢出、logits 由全
NaN 变为基本健康、输出由空串变为非空。两条是**互补依赖**关系，缺一不可。

**改动（都可逆、默认关）：**
- **GPU 插件（§7-1，让 fused MoE 接受 f32 layout）** —— 6 处，f16 路径完全不变（全是 additive）：
  - `moe_router_fused_opt.hpp` `validate_impl`：`supported_types` 加 `f32`。
  - `moe_3gemm_swiglu_opt.hpp` `validate_impl`：激活 `supported_types` 加 `f32`（权重/scale 仍 f16/int4）。
  - `moe_scatter_reduction_opt.hpp` `validate_impl`：`supported_types` 加 `f32`。
  - `moe_3gemm_swiglu_opt.cpp` 三个 grouped-prefill JIT 生成器（`PrefillGather`/`PrefillSwiglu`/
    `PrefillScatterReduce`）原本**硬编码 `"half"`**，改为按 `get_input_layout(0).data_type` 选
    `half|float`（并补 `MOE_DTYPE_SIZE`）。`.cl` kernel 本就有 `MOE_DTYPE_SIZE==4` 的 float 分支，
    无需改 kernel。canvas=256 > GEMV 阈值 → 实际走 grouped-prefill 路径，所以这三处是关键。
  - onednn grouped matmul 本就从 layout 取激活 dtype、权重保持 int4 → **不会重蹈 §6.4 eager-f32 OOM**。
- **建模（§7-2，残差主干标 f32）** —— `modeling_diffusion_gemma_text.cpp` 加 `DG_BACKBONE_F32` 开关：
  `detail::keep_f32()`（= 条件式 `disable_fp16_compression`）标住残差高速路：`h_attn`(1239) /
  `h_combined`(1260) / `h_ffn`(1262) / `out`(=out_prescale,1265) / 以及 layer_scalar 后的 `out`。
  fused MoE 仍是 **f16 孤岛**：`ops.cpp` 的 `moe3gemm_fused_compressed` 本来就 `input.to(f16)` →
  `MOECompressed(out_type=f16)` → `Convert(f32)`，标 f32 不碰它内部。

**关键发现 —— 为什么必须两条一起：**
- 只做全局 `DG_INFER_PRECISION=f32`（指望 §6.4"去掉 fused MoE 就能编译 f32"）**仍然失败**，但错误点
  从 `moerouterfused` 前移到了 **`moecompressed:MOECompressed_728 data_type:f32`**——`ConvertPrecision`
  把 f32 推到了 MoE 算子上，撞上 `moe_3gemm_swiglu_opt.hpp:57-64` 的 **scale 必须 f16** 检查。
  即：全局 f32 在**对抗**建模代码刻意造的 f16 MoE 孤岛，方向错。
- 正解是 §7-2：**全局保持 f16**，只把残差高速路选择性标 f32。但 router_input = 原始 `h_attn`（现 f32），
  所以**必须先有 §7-1 的 router f32 修复**才能编译——这正是 §5/§6.2 注释里警告的"标 f32 会再触发
  MoERouterFused f32-layout 错"的死结，§7-1 把它解开了。

**结果（`--max-denoising-steps 1`，dump 开）：**

| 指标 | 基线 f16 | **DG_BACKBONE_F32（本次）** |
|---|---|---|
| encoder | clean | **no-overflow**（max 1941） |
| decoder NaN | **67108864（全 256 位）** | **2359296（仅 9 位）** |
| max_logit | **−inf** | **37.5（健康，=softcap30/temp）** |
| mean_entropy | 0 | **10.47** |
| accepted | 0 | **10/256** |
| 输出 | 空串 | **非空** |

- **编译成功**：encoder + decoder 全 30 层 GPU 编译均无 layout 错（f32 主干 + f16 MoE 孤岛 + f32 router
  共存）。代价：f32 主干使 reorder/convert 增多、激活 buffer 翻倍，**全深度 decoder 编译显著变慢**
  （~190s 纯编译 + 大量 GPU 换页，墙钟可达 20–35min；进程在涨内存即正常推进，不是 hang）。
- **f32 高速路阻断了扩散**：8 层截断与全 30 层结果**完全一致**（同样 `dbg.072.l7.out_prescale`、同样
  9 位 NaN）。基线里 l7 的溢出会在 8–29 层间扩散到全部 256 位；现在 f32 主干把它**钉死在原始 9 位**，
  247/256 位变健康。这直接证明 §7-2 的主干 f32 承载是对的。

**仍残留的 9 位 NaN（待收口）：** `FIRST_NONFINITE=l7.out_prescale`，但其上游 tap（`h_attn`/`h1`/`h2`/
`h_ffn`）的 `max(|·|)` **都是有限值（≤2692）**——两个有限 f32 相加不可能成 NaN，所以 NaN 必从**没挂在
f32 高速路上的 f16 子段**（l7 是首个 `full_attention` 层，怀疑注意力 SDPA 内部，或 MLP/MoE 分支中间量
在个别 position 越界后被 norm 卷进残差）按位混入。

**⚠️ 全量 e2e（`--output-tokens 256`，48 去噪步）仍是空串。** 单步（`--max-denoising-steps 1`）只剩 9 位
NaN、247 位健康；但全量去噪**第一步（step 48）就是全 256 位 NaN**（`max_logit=-inf`）。差异在**时间步/
canvas 调度**：e2e 首步的 `t` 噪声水平下，l7 的 9 位 leak 经下游 23 层 matmul 当步就扩散到全部 256 位
（step 47 首溢点已前移到 `l8.out_prescale`）。即：主干 f32 修好了"残差累加溢出"，但 **l7 内部那个 f16
子段按位产生的 9 位 NaN** 仍是 e2e 的拦路虎，必须收口。下一步：① `DG_DUMP_MAXABS=full` 逐 tap（含
q/k/v_norm、rope、sdpa 细 tap）定位 l7 内部首个非有限子节点；② f32 标记已**加宽**到
`attn_post`/`mlp_out`/`h1`/`h2`（每层残差计算里剩余的 norm 输出子段）；③ 若仍漏，需把 l7 的注意力
SDPA / `full_attention`（head_dim=512、scaling=1.0）内部也纳入 f32 保护。

工具：sample 加 `DG_BACKBONE_F32=1`（建模侧，可逆默认关）；GPU 插件 6 处改动是 additive，f16 默认路径零影响。

**🔬 逐 tap 定位（`DG_DUMP_MAXABS=full`，8 层，加宽标记后）—— 残留 NaN 不是"幅值溢出"：**
l7 全部命名 tap 都**有限且很小**：`attn_out=52.5`、`sdpa=15.3`、`h_attn=42.9`、`h1=2076`、`h2=599`、
`h_ffn=29.6`；全图 `max_finite` 仅 **2778**（@l6.h1），离 65504 差得远。但 `l7.out_prescale = inf`
（`= h_attn + h_ffn`）。**两个 ~43 和 ~30 的有限 f32 不可能相加得 inf** —— 说明残留的 9 位 NaN
**根本不是幅值越过 f16 上限**，而是在 f32-marked `out` 节点的 **convert 边界**上凭空产生的 inf/NaN
（即便 `h_attn/h_ffn/h1/h2` 都已标 f32 且读数有限，且加宽标记后依旧）。l6 完全正常、l7 才坏；l7 是首个
`full_attention` 层（head_dim=512、scaling=1.0）。**这把"gamma 驱动溢出"假说从这 9 位上排除了** ——
它是 GPU 在 f16↔f32 微边界处的 reorder/miscompile 伪影，而非数值越界。下一步需深入 GPU 插件查
`out` 这个 Add 周围 convert/reorder 的实际精度与缓冲，而非继续加宽建模侧 f32 标记（加宽到
`attn_post/mlp_out/h1/h2` 已证明对这 9 位无效）。

---

## 6.8 ❌ convert-边界假说 + 内存复用假说 **双双被实验推翻**（2026-06-18）

§6.7 末尾"inf 生于 f32 convert 边界"的结论**是错的**。两个对照实验把精度方向彻底排除：

**对照 A — 纯 f16（不设 `DG_BACKBONE_F32`，全图无任何 f16↔f32 边界）：** 8 层 full-dump 下
`l7.out_prescale` **仍是 inf**（h_attn=43.0、h_ffn=30.1、nan=1835008、输出乱码）。⇒ inf 与 f32 标记
无关，也不存在 convert 边界。三条铁证：(1) 纯 f16 无边界也 inf；(2) `DG_BACKBONE_F32` 对 l7 几乎
**毫无改变**（f32 跑 43/30→inf vs 纯 f16 跑 43/30→inf，几乎一模一样），所谓"f32 高速路阻断扩散"只是
全-30-层的假象，8 层即证伪；(3) **encoder 用完全相同的残差 add 结构**（`out=h_attn+h_ffn`，
`apply_layer_blocks_after_attn` 被 encoder/decoder 共用）、且 l7 输入**更大**（h_attn=81.7、h_ffn=46.3）
却**有限**（out_prescale=127.9），而 **decoder** 用**更小**输入（43+30）反而得 inf。

**对照 B — 内存复用/池：** 纯 f16 8 层分别跑 `OV_GPU_DISABLE_MEMORY_REUSE=1` 与 `OV_GPU_ENABLE_MEMORY_POOL=0`，
两者产生**逐字节相同的 inf**（nan=1835008、argmax 完全一致）。⇒ 缓冲复用/内存池 aliasing **也不是**原因。

**🔑 核心矛盾（已铁证）：** NaN-fold 的 maxabs tap 证明 `max|h_attn|≤43`、`max|h_ffn|≤30`（任一元素若是
NaN 会 fold 成 +inf 显出来），而 `out=h_attn+h_ffn=inf`。两个各自被 ±43/±30 界定的张量做正确逐元素相加
**不可能**超过 ±73。⇒ decoder l7 残差 add 算子**没有在逐元素地加 tap 测到的那两块缓冲** —— 要么读了
**越界/错 stride/错 shape** 的内存，要么 tap 读的物理缓冲 ≠ add 消费的缓冲。这是**纯结构/内存 bug，与精度无关**。

**🔬 编译图 ground truth（`OV_GPU_DUMP_GRAPHS_PATH`，decoder=program 3，`build_implementations`）：**
- l7 残差 add `add:Add_46659`：kernel **`generic_eltwise_ref`**（= `eltwise_kernel_ref` / `generic_eltwise_ref.cl`），
  输出 `f32:bfyx:?x?x2816`（**动态 shape**），**未被 fuse**（standalone）。
- input0 = `add:Add_46549`（h_attn）= **f32**；input1 = `rms:Multiply_46658`（h_ffn）= **f16**。
  即 `DG_BACKBONE_F32` 下这个 add 是真正的 f32+f16 混合输入 → `generic_eltwise_ref`。但纯 f16 下它是 f16+f16
  也照样 inf，故混合 dtype 不是触发因素。
- ⚠️ **h_ffn 的 `post_feedforward_layernorm` 被降为内部融合算子 `ie_internal_opset::RMS`**（kernel
  `rms_gpu_bfyx_opt`，输出 f16，eps=1.5625e-08）—— 建模侧手写的 `pow→mean→rsqrt` + `protect_rms_variance`
  f32 标记被这步融合**抹掉**。但已查 `RMSKernelBase::GetAccumulatorType`：f16/f32 输入**累加器恒为 f32**
  （rms_kernel_base.cpp:88-98），故 RMS 内部不会 f16 溢出 —— RMS-内部-溢出假说也排除。
- 已确认 eltwise 内核本身 **dtype 安全**：`GetAccumulatorType` 任一输入是 f32 即用 f32 累加
  （eltwise_kernel_base.cpp:102-107），各输入按各自 typed 指针读（:437）。

**结论：bug 在 decoder program 对这个 `generic_eltwise_ref` 动态-shape add 的内存/索引布局上**（encoder 同结构正常 ⇒
差异只在有状态 decoder：KV-cache ReadValue/Assign、self-cond 输入、以及随之不同的 shape/buffer/memory-dependency 走法）。

**下一步（二选一，已在跑/已规划）：**
- (1) 免重编：`OV_GPU_DUMP_TENSORS` + `OV_GPU_DUMP_LAYER_NAMES` 读 `Add_46659`/`Add_46549`/`Multiply_46658`
  的**实际字节**，直接判定"tap 撒谎"还是"add 读越界"。⚠️ 注意：tensor dump 在 genai 内部 decoder 网络上
  **可能不触发**（dump 文件为 0），需先用 `OV_VERBOSE` 确认运行期 primitive id，或确认 dump 配置是否随
  genai 编译路径传入。
- (2) 需重编的 modeling clamp 实验：`out = clamp(h_attn,±1e4)+clamp(h_ffn,±1e4)`。仍 inf ⇒ add 读的不是
  它被声明的输入 ⇒ 坐实结构/越界；变有限 ⇒ tap 漏掉了某个巨值。

**※ §6.7 的 6 处 GPU-plugin MoE f32 改动 + `DG_BACKBONE_F32` 建模 knob 仍是 additive/无害，但都不修这个 bug，
找到真正内存 bug 后可回退。** 全部精度方向调查（ASF / backbone-f32 / MoE-f32）都是在追一个精度的红鲱鱼。

---

## 6.9 ✅✅ 根因坐实：残差 add 的 **结构性 GPU-plugin miscompile**（动态-shape decoder），与值/精度无关（2026-06-18）

建模侧加 `DG_CLAMP_RESIDUAL` 探针：`out = Clamp(h_attn,-hi,+hi) + Clamp(h_ffn,-hi,+hi)`，bound 由
`DG_CLAMP_HI` 配（Clamp **不标 f32**，忠实测纯 f16 路径）。纯 f16、8 层、1 step、full dump 扫描：

| DG_CLAMP_HI | decoder l7.out_prescale | nan |
|---|---|---|
| 无（baseline） | **inf** | 1835008 |
| 1e4 | **61.7** | **0** |
| 100 | **63.3** | **0** |
| **1e30** | **61.7** | **0** |

**`1e30` 是 TRUE value no-op**（残差流最大才 ~2778，离 1e30 差 26 个数量级，clamp 不可能改任何值），
**却照样把 inf 和全部 NaN 消掉**。⟹ **铁证：值上限无关紧要 —— 仅仅在残差生产者
（h_attn=`add:Add_46549`、h_ffn=融合 `rms:Multiply_46658` f16）与 `generic_eltwise_ref` add 之间插入一个
全新 elementwise `Clamp` 算子**，就强制出一块干净中间缓冲 / 不同的 kernel-arg 接线，**绕开了 miscompile**。
裸 add 在动态 `?x?x2816` 下读了**错内存（越界/错 stride/错 offset）**。encoder 用完全相同建模代码、相同
`generic_eltwise_ref` kernel，但其 program 编译方式不同（无状态、无 KV-cache ReadValue/Assign、无 self-cond），
故同一个 add 在 encoder 算对、在 decoder 算错。

**这坐实了 §6.8 的结构假说，并排除了"tap 漏掉巨值"的分支（§6.8 选项 2 的第二种可能）。**

⚠️ **NaN 没了但单步输出仍不正常**（accepted=1/256、生成文本空/乱）—— clamp 修掉了 inf，但单步解码质量还差；
需跑全 e2e（48 步）看 clamp 单独能否产出真正文本。**clamp 是 workaround 不是 fix。**

**真正的修复在 GPU 插件**：查清 decoder 的 `generic_eltwise_ref` add 为何在动态 shape 下 miscompile。
嫌疑：eltwise_ref 在动态 `?x?x2816` 下的 offset/index 计算（`GET_INDEX`/`OUTPUT_SIZES`/`ELTWISE_NO_PITCH_SAME_DIMS`
分支），叠加 input1 是 f16 融合-RMS 输出。**下一步**：(a) 确认 clamp 在全 e2e 下成立；(b) diff encoder vs
decoder 这个 add 的 kernel args / jit（GET_INDEX、OUTPUT_SIZES、in/out pitch），定位并修复 eltwise_ref 动态-shape
内存 bug —— 修好后 clamp + §6.7 全部精度改动都可回退。

`DG_CLAMP_RESIDUAL` + `DG_CLAMP_HI` knob 保留在建模里（默认关、additive）。

> ⚠️ **§6.9 的结论（"结构性 miscompile / 值无关"）后来被 §6.12 推翻**：clamp 之所以"修好"，是因为
> clamp 的输入挂着 f32 tap，Clamp 阻断了对这些 f32 输入的 fp16-compression，等于把残差边提成 f32 ——
> 又是 tap 制造的精度伪影，不是结构修复。保留本节作为"诊断件本身污染测量"的典型反面教材。

---

## 6.10 split-clamp + working-vs-broken JIT diff：缩到极窄但需运行期状态（2026-06-18）

给 sample 加 `DG_CLAMP_SIDE`（attn|ffn|both）只 clamp 一侧。结果：clamp **both** → out_prescale 有限(61.7)；
clamp **attn-only → 仍 inf**；clamp **ffn-only → 仍 inf**；不 clamp → inf。⟹ **推翻"单操作数 over-read"**
（必须两侧都插新算子）。再 dump WORKING(clamp-both) vs BROKEN(no-clamp) 的 baked `.cl`、相同 fusion-on 拓扑做 diff：
残差区每个 kernel（standalone-when-nofuse 的 `generic_eltwise_ref` ADD `[b,seq,2816,1]` bfyx、连续、PAD=0、
SAFE index；以及 `rms_gpu_bfyx_opt` 生产者）**逐字节相同**。唯一 JIT 差异是 (a) working 多 8 个 `activation_ref`
（Clamp 物化）和 (b) broken 才有的 attention-only `concatenation_gpu_simple_ref` + `select_gpu_ref`（KV-cache
副作用，红鲱鱼）。⟹ **bug 不在任何 kernel 的生成代码**，而在写进 `shape_info[]` 的运行期值或残差 add 的
缓冲分配/绑定 —— 编译期 `.cl` dump 看不见。**当时的（错误）结论**：in-place/buffer-reuse aliasing；下一步规划
在 `eltwise_impl::get_arguments` 加 env-gated probe dump 运行期 in/out shape + buffer ptr。

---

## 6.11 ⚠️⚠️ 重大反转：`DG_DUMP_MAXABS` 的 tap 本身改变了计算（2026-06-18）

**§6.7–§6.10 的每一个"`l7.out_prescale` / `h_attn=43` / clamp 修复 / 7-of-256"测量，都是在一张
被部分提成 f32 的图上测的，不是真实推理。** 给 GPU 插件加 `DG_ELTWISE_PROBE`（`eltwise.cpp::get_arguments`：
对任何输出末维==2816 的 eltwise，dump 每个操作数的运行期 ptr/bytes/count/layout）。

PROBE 结果（**带 tap**）：decoder 残差 add 跑成 **f32(in0) + f16(in1) → f32(out)**，**无 buffer aliasing、
无 over-read**（f16 缓冲字节数减半但元素数相同，正确）。那些 f32 全来自 maxabs tap：`tap_maxabs` 插
`x.to(f32)` + `disable_fp16_compression`，`AlignMixedFP32FP16Types`/`ConvertPrecision` 于是把被 tap 的残差边
带成 f32 → 一个在未-tap 推理里**根本不存在**的 f32+f16 混合 add。

**决定性对照（无-tap）**：跑 NO-TAP（不设 `DG_DUMP_MAXABS`），8 层、1 step —— `[denoise] step 1` 的
logit 统计（tap 无关，从真实 logits 来）= **nan=67108864（全部 256 位），max_logit=−inf，argmax 全 0**，
**远比 tapped 的 7/256 严重**。⟹ tap 不是被动测量 bug，它的 f32 提升在**部分掩盖** bug（7/256 而非 256/256）。
所以"out_prescale=inf 7 位"和整条 §6.7–§6.10 链描述的是**被 tap 的图**的行为，是另一份计算。clamp"修复"
也只对被 tap 的图成立。

**教训**：任何 f32-marked 诊断（tap、`keep_f32`、`DG_BACKBONE_F32`）都会扰动正被调查的精度路径；要用
插件侧 `DG_ELTWISE_PROBE`（读缓冲、不加算子）和 tap-无关的 denoise nan 计数。

---

## 6.12 ✅✅✅ tapless 坐实真因 = 原始 f16 残差主干溢出（第 21 行最初判断），非任何 miscompile（2026-06-18）

增强 `DG_ELTWISE_PROBE`：mem_lock + host 扫描每个 2816 eltwise 的**输入**缓冲的 maxabs/nan/inf，无-tap 跑
（真实纯 f16；残差 add 被 fuse 掉，所以只有 RMSNorm 的 `multiply:Multiply_*` 露出来 —— 其 in0 就是喂给每个
norm 的残差主干值）。decoder 逐层 in0 maxabs 轨迹（8L、1 step、tapless）：

```
1138 → 1461 → 3934 → 4484 → 1.56e4 → 1.55e4 → 3.57e4 → 3.77e4 → 4.71e4 → 6.40e4 → 5.90e4(inf=5) → maxabs=0 nan=720896(全部)
```

（in1 = 每元素 rms_inv 尺度，逐层正确缩小 0.069→0.0004。）原始残差流单调增长、约第 6→7 层越过 65504，
首个 inf 出现，随后 RMSNorm `rsqrt(mean(x²))`：含 inf 的 mean=inf，rsqrt(inf)=0，0×inf=NaN，扩散到全部
720896。**这正是第 21 行"CORRECTED ROOT CAUSE：溢出是 Gemma 残差主干、约第 7 层越过 65504"。**

§6.8–§6.11 的"convert-边界推翻 / 结构 miscompile / clamp-both-修复 / 运行期缓冲不匹配"整条链，**全是
f32 maxabs tap（`DG_DUMP_MAXABS`）的伪影**：tap 把残差边提成 f32、抬高溢出上限，所以 tapped run 只显示
7/256 NaN。"clamp 修复"是 Clamp 阻断了它那些 f32-tap-提升的输入的 f16-compression（同样的掩盖）。
净结论：真 bug = 纯 f16 残差高速路动态范围溢出（HF 跑 bf16 永不溢）。

---

## 6.13 修复进展：f32 高速路生效，缺口在 norm 输出（2026-06-18）

`DG_BACKBONE_F32` 的 f32 高速路**逐层有效**：残差流在 f32 下涨到 1.05e5(>65504) 全程无溢出（probe：in0 f32
max 9→1431→2692→...→1.049e5，nan=0）。唯一漏点：最后一个 decoder add `Add_70044 out=f32, in0 f32(1.052e5) ok,
**in1 f16 max=6.544e4 with inf=8**` —— in1 即 h_ffn（`post_feedforward_layernorm` 输出），**仍是 f16 并在其自身
下采时内部溢出**（8 个元素超 65504）。加了 `DiffusionGemmaRMSNorm::forward(x, keep_output_f32=true)` 重载（跳过
`.to(f16)`）并用在 4 个高速路 norm 上，但 probe 显示 in1 仍 f16（量级从 65056→65440，有 SOME 效果但 GPU 精度
pass 仍下采）。

---

## 6.14 全-30 e2e 仍全 256 NaN，缺口精确定位到融合后的 norm 输出（2026-06-18）

全-30 `DG_BACKBONE_F32` + `keep_output_f32`、tapless、带 probe。f32 **高速路完美**：in0 涨到 **1.052e5** 全程
nan=0/inf=0。90 个 decoder 2816-add：59 个 f32-out、31 个 f16-out。首个失败 = `Add_70044 out=f32, in0 f32(1.052e5) ok,
**in1 f16 max=6.544e4 inf=8**`。⟹ 残差 add 输出正确是 f32、in0 高速路是 f32，但 **in1（h_ffn，post_feedforward_layernorm
的输出）仍是 f16 并内部溢出**。`keep_output_f32`（返回 f32 `scaled` + `disable_fp16_compression`）**没能让
h_ffn 保持 f32**。当时的诊断：norm-输出 `scaled` 上的 `DisableFP16Compression` rt_info 在 GPU 上未被 honored。

---

## 6.15 🎯 精确定位修复点：RMSFusion 丢标记（2026-06-18）

norm 输出之所以保持 f16，是因为 GPU `RMSFusion`（`transformations_pipeline.cpp:660`）在 ConvertPrecision **之前**
把手写 RMSNorm 子图融成单个 `ov::op::internal::RMS` 算子，途中用 `ov::copy_runtime_info` 拷贝时**丢掉了**建模侧
`keep_output_f32`/`keep_f32(scaled)` 打在中间 multiply 上的 `DisableFP16Compression` 标记（因为
`DisableFP16Compression::is_copyable()` 返回 **false**，见 `disable_fp16_compression.hpp:41`）。插件原有补救
`DisableFP16CompForGemma3RMSPattern`（`disable_fp16_comp_rms.cpp`）只匹配 Gemma3 的**串行**残差且每节点
`type_matches(f32)`，**匹配不上 DiffusionGemma 的并行 block**。**当时（部分正确）的计划**：写一个 DiffusionGemma
专用 DisableFP16Comp pattern pass，把喂给 f32 残差 add 的 RMS 标 `disable_fp16_compression`。

---

## 6.16 ❌ 重标融合后的 RMS 算子 **无效**（DG_PASS_DEBUG 坐实，2026-06-18）

按 §6.15 写了 `DisableFP16CompForDiffusionGemmaRMSPattern`（anchor 在 `RMS`、walk 到被标记的残差 `Add`、
重新 `disable_fp16_compression`）。加 `DG_PASS_DEBUG` 打印每个被访问 RMS 的 out 类型/consumer/标记。**ground truth**：

```
[DG_PASS] RMS Multiply_96678 out=f32 already_marked=0 | consumer=Add(Add_96679,marked=1) feeds_marked=1
```

`Multiply_96678` 正是喂给 probe 报错的那个 `Add_96679` 的 RMS，pass **把它标成了 f32**（`feeds_marked=1`，
pass-时 out=f32）—— **可 probe 仍显示该 add 的 in1=f16**。⟹ **决定性新事实：标记融合后的
`ov::op::internal::RMS` 算子为 f32，并不产生 f32 输出缓冲** —— RMS GPU kernel（`rms_gpu_bfyx_opt`）无视算子声明
的输出类型，强行把输出下采成 f16。**重标融合算子是错的杠杆。**

---

## 6.17 ✅✅✅✅ 解决并端到端验证（2026-06-18）

**起作用的杠杆**：不要重标融合后的算子，而是**让被标记的高速路 norm 保持不被融合** —— 它们就以建模侧已标记的
f32 分解原语存在，跟那些本来就守住 f32 标记的残差 Add（generic eltwise）同一机制。建模侧的残差 Add 标记本就
有效（probe：in0=f32），只有 RMS 丢了标记。

**实现（一行）**：扩展 `transformations_pipeline.cpp` 里既有的 `pass_config->set_callback<ov::pass::RMSFusion>`
（带 `max_work_group_size` guard 的那个，~653 行），**最前面**加：

```cpp
if (ov::fp16_compression_is_disabled(root)) {
    return true;   // 跳过融合：这些被标记的高速路 norm 保持分解 f32 原语
}
```

`root` 是 gamma multiply = 建模侧的 `scaled` 节点，匹配时（融合丢标记之前）带着 `keep_f32` 的标记。补
`#include "transformations/rt_info/disable_fp16_compression.hpp"`。默认模型 no-op（没节点带这标记）。

**PROBE 确认**：每个 decoder 高速路 norm-输出 multiply 现在以 standalone `id=multiply:Multiply_* out=f32:1x256x2816`
出现（原来融合→f16）；残差 add 的 in0+in1 **都是 f32**；逐层 pre_feedforward norm（喂给刻意保留的 f16 MoE/MLP
孤岛、未标 keep_f32）正确保持 `out=f16`。

**端到端验证：**

| | 修复前 | **修复后（8-token verify）** | **修复后（256-token 终验，零 env）** |
|---|---|---|---|
| nan / inf | 全 256 NaN，inf=8 | **全程 0 / 0** | **全程 0 / 0** |
| max_logit | −inf | 36–73（健康） | 36–73 |
| denoise | entropy 0、accepted 0 | 收敛（step1 entropy 0.06、accepted 153/256） | 收敛 |
| 输出 | 空串 | "The sky appears blue because of a phenomenon" | **连贯多段**（见下），`FINAL_EXIT=0` |

256-token 终验（默认 build、**无任何 DG env 变量**、全部诊断已移除）：

```
--- Generated text ---
The short answer the blue sky is blue due to a phenomenon called **Rayleigh scattering.**
Here is the step-by-step breakdown of why this happens:
### 1. Sunlight contains all colors  ...
### 2. The atmosphere acts as a filter  ...
```

**可上线的最终改动集：**
1. **genai `modeling_diffusion_gemma_text.cpp`**：`backbone_f32_enabled()` 改为 `return true;`（原 env gate；
   按需可保留 `DG_BACKBONE_F32` 仅作 debug 覆盖）；`keep_f32()` 标住高速路节点；`RMSNorm::forward(x, keep_output_f32)`
   重载对 4 个高速路 norm 返回 f32 `scaled`。**移除**了 `DG_CLAMP_*`/`DG_CLAMP_SIDE`/`clamp_residual*` 诊断
   （`DG_DUMP_MAXABS` tap 是早先既有工具，保留）。
2. **GPU 插件 `transformations_pipeline.cpp`**：RMSFusion skip-callback 一行 + rt_info 头。
3. **移除**：`disable_fp16_comp_dg_rms.{cpp,hpp}`（§6.16 那个无效的 pattern pass）及其注册；`DG_ELTWISE_PROBE`
   / `DG_PASS_DEBUG` 插桩；`eltwise.cpp` 恢复干净（删掉 `get_arguments` override + probe 相关 include）。
4. **回退（§6.18）**：§6.7 的 6 处 MoE-f32 改动 —— `moe_router_fused_opt.hpp` / `moe_3gemm_swiglu_opt.{hpp,cpp}`
   / `moe_scatter_reduction_opt.hpp` 全部 revert 回上游 baseline（经验证不需要，见 §6.18）。

满足硬约束：f16 仍是全局精度，只有薄薄的残差高速路 + 喂它的 norm 走 f32。

**本节核心教训（给未来的人）**：
- f32 标记要落到**实际产出缓冲的算子**上才有效。标记一个会被**后续 pass 融合掉**的算子（如手写 RMSNorm
  被 `RMSFusion` 吞）→ 标记随 `copy_runtime_info` 丢失（`DisableFP16Compression` 不可拷贝）。
- 即使标记**幸存**到融合后的算子上，**融合算子的 GPU kernel 也可能无视声明的输出精度**（`rms_gpu_bfyx_opt`
  恒输出 f16）。所以"keep f32"的可靠做法是**阻止融合**（让算子保持分解原语），而不是给融合算子贴标记。
- 用 **tapless 工具**（插件侧读缓冲的 probe + pass 侧 `DG_PASS_DEBUG`）取得 ground truth；任何往图里插
  f32 算子的诊断都会污染精度路径（§6.11/§6.12 的惨痛教训）。

---

## 6.18 ✅ MoE-f32 插件改动（§6.7 的 6 处）经验证 **不需要** → 修复缩为两件套（2026-06-18）

§6.17 的 SHIPPABLE FIX 当时仍把"§6.7 的 6 处 MoE-f32 插件改动"列为第 3 件套（理由：router_input=f32 `h_attn`）。
做对照验证后**证伪**了这条依赖：把 6 处全部 revert（`moe_router_fused_opt.hpp` / `moe_3gemm_swiglu_opt.{hpp,cpp}`
/ `moe_scatter_reduction_opt.hpp` 的 `supported_types += f32` 与 grouped-prefill JIT 的 `half→half|float`），
重编 GPU 插件，e2e（无 env、backbone-f32 始终开）：

| | 带 6 处 MoE-f32 改动（§6.17） | **revert 后（本节）** |
|---|---|---|
| encoder/decoder 编译 | OK | **OK（无 `moerouterfused/moe3gemm data_type:f32` layout 错）** |
| nan / inf | 0 / 0 | **0 / 0** |
| step-48 统计 | entropy 4.5278 / max_logit 36.2305 / argmax 818 818 506… | **逐字节相同** |
| 输出 | 非空连贯文本 | **非空连贯文本（"The short answer the blue sky is blue…"）** |

⟹ **6 处 MoE-f32 改动在最终修复下是 dead code，可删**。**根因**：最终的 RMSFusion-skip 修复只把残差高速路
（`h_attn/h_combined/h_ffn/out` + 4 个喂它的 norm）标 `keep_f32`。MoE block 的 router/experts 预处理
（`DiffusionGemmaMoEBlock::forward_fused` 里的 `router_2d`/`experts_2d` → `rms_norm_no_scale` → matmul）
**从不** keep_f32，且 `ops::moe3gemm_fused_compressed` 显式 `input.to(f16)` —— 所以 `ConvertPrecision` 把整条
MoE 路径下采回 f16，`MoERouterFused`/`MOE3Gemm`/`scatter` 收到的是 **f16**，f32 的 `validate_impl` 分支根本不触发。
§6.7 那条"router_input=f32 → 必须让 MoE 接受 f32"的死结，是**早期更宽的 f32 标记策略**下的真问题，对最终的最小
标记策略不成立。

> ⚠️ 注意：`moe_3gemm_swiglu_opt.cpp` 的**上游 baseline 本来就**在非-prefill 的 GEMV 内核（~90/347/404 行）有
> `moe_is_f16 ? "half" : "float"`，那是既有代码、**不是**我们的改动；我们 revert 的 6 处是 grouped-prefill 的
> 221/264/312 行 + 3 个 `validate_impl` 的 `+= f32`。这 4 个文件现已回到上游 baseline。

**→ 最终修复 = 两件套**（见下方 §7 已更新）：(1) genai 残差高速路 f32（始终开）；(2) GPU 插件 RMSFusion skip 一行。

---

## 7. 最终修复（§6.17 落地、§6.18 收敛为两件套，已端到端验证）

**✅ 完整修复 = 两件套，缺一不可：**

1. **⭐ 残差主干标 f32（建模侧，§6.7/§6.13）**。`detail::keep_f32()`（= `disable_fp16_compression`）标住
   残差高速路 `h_attn/h_combined/h_ffn/out`，并对喂高速路的 4 个 norm 用 `forward(x, keep_output_f32=true)`
   返回 f32。`backbone_f32_enabled()` 已改为 **`return true;`**（DiffusionGemma 始终开启，GPU 才能开箱即用；
   `DG_BACKBONE_F32` env 可作 debug 覆盖）。fused MoE 天然是 f16 孤岛（`ops.cpp`：`input.to(f16)`→
   `MOECompressed(f16)`→`Convert(f32)`），标 f32 不碰它内部。
2. **⭐⭐ GPU 插件让 RMSFusion 跳过被标记的高速路 norm（§6.17，本次会话的关键发现）**。在
   `transformations_pipeline.cpp` 的 `RMSFusion` skip-callback 最前面加
   `if (ov::fp16_compression_is_disabled(root)) return true;`。**为什么必须这样**：第 1 条把 norm 输出标 f32
   后，GPU `RMSFusion` 会把手写 RMSNorm 融成 `ov::op::internal::RMS` 并丢掉不可拷贝的标记（§6.15）；即使
   重新给融合算子贴标记也**无效** —— RMS GPU kernel 无视声明类型恒输出 f16（§6.16）。唯一可靠做法是**阻止
   融合**，让这些 norm 保持分解后的 f32 原语（跟残差 Add 同机制）。默认模型 no-op。
   （需要 `#include "transformations/rt_info/disable_fp16_compression.hpp"`。）

**~~第 3 件套~~ 让 GPU MoERouterFused/3gemm/scatter 接受 f32 layout（§6.7）—— 经 §6.18 验证 *不需要*，已删。**
最终修复下只有残差高速路 + 喂它的 norm 标 f32；MoE 的 router/experts 预处理从不 keep_f32 且 `moe3gemm_fused_compressed`
显式 `input.to(f16)`，所以 ConvertPrecision 把整条 MoE 路径下采回 f16，MoE 算子收到 f16，f32 layout 分支永不触发。
revert 这 6 处后 e2e 编译/输出/逐步统计与带改动时**逐字节相同**（§6.18）。

**已被证伪、不要再走的路（保留为反面教材）：**
- 全局 `DG_INFER_PRECISION=f32`：对抗建模刻意造的 f16 MoE 孤岛，错误从 router 前移到 `MOECompressed` 的
  scale-必须-f16 检查（§6.7）。
- `ACTIVATIONS_SCALE_FACTOR`（任何值）：`×S` 上提在残差 add 处被保留，主干幅值对 S 不敏感，还把 encoder
  弄溢出（§6.6）。
- INT4-asym 量化：同深度同输入对照，换 F32 MoE 后逐位一致溢出（§6.5）。
- "结构性 eltwise miscompile / clamp 修复 / 内存复用 aliasing"（§6.8–§6.11）：全是 `DG_DUMP_MAXABS` f32 tap
  的伪影（§6.12）。
- eager MoE + 全局 f32：显存 OOM（eager 平铺 `[64,256,2816]` f32 ≈1.8GB/层，§6.4）。

> **复现/调试工具**（sample + 插件，可逆）：`DG_DUMP_MAXABS=1`/`=full`（逐层 max-abs dump，⚠️ **会污染精度
> 路径**，仅作粗定位）；`DG_ELTWISE_PROBE`（**tapless**，插件侧读 2816 eltwise 缓冲的 dtype/maxabs/nan/inf，
> 不加图算子 —— 验证修复的正确工具）；`DG_PASS_DEBUG`（pass 侧打印 RMS 访问/consumer/标记）。**这三个都是
> 诊断件，不上线**；`DG_ELTWISE_PROBE`/`DG_PASS_DEBUG` 已从源码移除，`DG_DUMP_MAXABS` 是既有工具保留。
> 插件重编 + 部署：`cmake --build build\build_ov --config Release --target openvino_intel_gpu_plugin`，
> 把 `thirdparty\openvino\bin\intel64\Release\openvino_intel_gpu_plugin.dll` 拷到
> `build\install\runtime\bin\intel64\Release\`（sample 实际从这里加载）。⚠️ **新增/删除源文件后必须先
> `cmake .` 重新 configure**（CMake 用 `file(GLOB_RECURSE ...)` 无 `CONFIGURE_DEPENDS`，否则新 .cpp 不参与
> 编译会 LNK2019、删的 .cpp 仍被编译）。

---

## 8. 复现 / 调试速查

- **远程**：`sshpass -p 'openvino' ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null Local_Admin@10.239.132.229 "<cmd>"`
  - ⚠️ **不要用 bash `eval` 包 ssh 命令** —— `eval` 会吞掉 Windows 路径里的反斜杠
    （`dir C:\foo` 变成 `C:foo` → "File Not Found"）。直接 `sshpass ... ssh ... 'dir C:\foo'`。
  - 仓库根：`C:\Users\Local_Admin\river\ovmx\openvino.pipeline.mx`，genai 源在
    `thirdparty\openvino.genai\src\cpp\src\modeling\...`；模型在 `D:\chenhu\model\diffusiongemma-26B-A4B-it`。
  - cmd shell **会把 `|` 抢在 PowerShell 之前解析** → `Format-Table`/`ForEach-Object` 报
    "not recognized"。要用管道/cmdlet，把脚本写成 `.ps1` scp 过去，`powershell -File C:/x.ps1`
    （**用正斜杠**，反斜杠同样会被吞）。
  - cmd `set VAR=val && cmd` 会**把 `val` 后的尾随空格也算进环境变量**（`"full "`≠`"full"`）。
    用 `set "VAR=val" & cmd` 形式去掉尾随空格。
  - 源文件是 **LF** 行尾；scp 保字节，安全。过滤 GPU 噪声：
    `grep -avE "GPU_Debug|MemoryTracker|check_allocatable|memory swap"`。
- **逐层 max-abs dump（本次新增工具）**：环境变量 `DG_DUMP_MAXABS` 开启。
  - 实现：`modeling_diffusion_gemma_text.cpp` 的 `detail::tap_maxabs`/`maxabs_scalar` 给每层
    hidden 注册 `dbg.NNN.<site>` 标量输出 = `max(|x|)`，子图强制 f32 +
    `disable_fp16_compression`，并 **fold NaN→+inf**（GPU `ReduceMax` 忽略 NaN，必须
    `where(x!=x,+inf,|x|)` 才看得见 NaN）。打印在 sample 的 `print_dbg_taps`。
  - `DG_DUMP_MAXABS=1`：每次 infer 打 `[maxabs <label>] FIRST_NONFINITE=... max_finite=...@...`。
  - `DG_DUMP_MAXABS=full`：额外加注意力内部细 tap（q/k/v_norm、rope、sdpa；它们只有单一消费者，
    平时不挂以免破坏 SDPA/RoPE 融合）并逐 tap 打印全部值。
  - 默认关闭 → 生产图零开销、拓扑不变。
- **只编译 sample**（会连带重编 `openvino_genai_obj`，所以改 `modeling_diffusion_gemma_text.cpp`
  也会生效）：
  ```
  cd C:\Users\Local_Admin\river\ovmx\openvino.pipeline.mx
  cmake --build build --config Release --parallel 16 --target modeling_diffusion_gemma
  cmake --install ./build/ --config Release --prefix ./build/install
  ```
  重编前先 `taskkill /F /IM modeling_diffusion_gemma.exe`（否则 exe 被占用，链接报 LNK1104）。
- **快速迭代**：用 `--num-layers N`（如 6）和 `--max-denoising-steps 3` 截断，秒级 vs 分钟级；
  26B 全量在 34GB GPU 上会换页，编译 + 映射权重共需十几分钟。
- **关键诊断**：sample 的 `[denoise]` 行已直接打印 `nan` / `inf` / `max_logit` / `argmax`，
  是定位 NaN 的最快入口，无需额外插桩即可判断"溢出 vs logit 坍缩"。
