# 支持 diffusion_gemma MoE 的分析与结论

> 范围：在 openvino.mx（`/mnt/river/ovmx/openvino.pipeline.mx/thirdparty/openvino`，与 `/mnt/river/ovmx/openvino.mx` 同源）
> 与 openvino.genai.mx（`thirdparty/openvino.genai`）中支持 HuggingFace `diffusion_gemma` 的 MoE
> （router + 专家计算）所需的改动。
> 日期：2026-06-16

---

## 1. diffusion_gemma MoE 的关键特征

来源：transformers `main` 分支 `src/transformers/models/diffusion_gemma/`
（`DiffusionGemmaTextRouter`、`DiffusionGemmaTextExperts`）。

### Router（门控）
- 输入先经 **无 scale 的 RMSNorm**，再乘 learned `scale` 向量与常数 `hidden_size**-0.5`，
  然后 `Linear(hidden, num_experts, bias=False)` 得到 logits。
- **softmax 在 top-k 之前**（fp32，over all experts），再 `topk`，
  再 `top_k_weights /= sum`（**无条件归一化**，无 `norm_topk_prob` 开关）。
- 归一化之后 **`top_k_weights *= per_expert_scale[top_k_index]`**
  —— 一个 learned 的每-expert 标量向量，按选中 index gather 后乘到 routing weight 上。
- **无 sigmoid，无 DeepSeek 式 additive routing bias，无 expert group（n_group/topk_group）**。

### 专家（Experts）
- `gate_up_proj` 是 **融合权重** `[E, 2*inter, hidden]`，
  forward 里 `.chunk(2, dim=-1)` 切成 **连续两半** gate / up（**非交错**）。
- 激活是 `ACT2FN["gelu_pytorch_tanh"]` → **GEGLU(tanh)**：`gelu(gate) * up`，再 `down_proj`。
- 即 **2-GEMM（融合 gate_up + down）+ GEGLU**，**无 bias、无 clamp**。

### 层结构与 diffusion 特性
- 一个 **dense MLP 与 MoE 并行求和**（dense 分支 = 事实上的 shared expert，
  但在 experts 模块**之外**，且各自带独立 layernorm）。
- diffusion 特性（双向 attention、timestep / self-conditioning）**不改变 MoE 本身**，
  与本分析正交。

### 关键配置参数名（与标准 MoE 不同，易踩坑）
- `num_experts`
- `top_k_experts`（**不是** `num_experts_per_tok` / `top_k`）
- `moe_intermediate_size`（每专家 FFN 中间维）
- `hidden_activation`（**不是** `hidden_act`），默认 `"gelu_pytorch_tanh"`

---

## 2. openvino.mx 现有能力（已支持）

| 能力 | 现状 |
|---|---|
| `MOE` / `MOECompressed` / `MOE3GemmFusedCompressed` op | 已有，含 `GEMM2_BIAS_SWIGLU_CLAMP` 与 `GEMM3_SWIGLU` 两种 expert_type |
| 激活类型 | `SWIGLU` / `GEGLU_TANH` / `GEGLU_ERF` 均**已在 GPU kernel 实现**（`moe_3gemm_swiglu_fuse.cl:254`）→ **gelu_pytorch_tanh 已覆盖** |
| Router: softmax-before-topk + 归一化 | `fuse_moe_experts.cpp:206` 匹配 `Softmax(axis)→TopK→OneHot`，再 `ReduceSum→Divide` 归一化 → 与 diffusion_gemma 的 router 主干一致 |
| Routing 类型 | `RoutingType::{SOFTMAX, SIGMOID_BIAS}`；compressed 的 SOFTMAX 路径**有 per-expert scale（Gather+Multiply）折叠支持** |
| GPU Router 融合 | `FuseMOE3GemmCompressed`（`fuse_moe_3gemm_compressed.cpp`）把 router 子图重新吸收进 `MOE3GemmFusedCompressed`；该 pass 的 softmax 分支**已有 optional 的 per_expert_scale Gather + Multiply 节点**（L62-67），并把它**折叠进 down(w2) scale**（L207-229） |
| shared expert | GPU kernel 有 `num_shared_expert>0` 路径（语义：带 scalar-gate 的共享 expert，**非** diffusion_gemma 的独立 dense MLP） |

---

## 3. 缺口分析（要支持 diffusion_gemma 所缺）

### 缺口 1【主要】专家侧：缺「融合 gate_up（连续 chunk）+ GEGLU」的融合 pattern
现有两条专家融合路径都对不上 diffusion_gemma 的 IR：
- **3-GEMM tiled pattern**（`convert_tiled_moe_block_to_gather_matmuls.cpp:139`）
  要求 gate 与 up 是**两个独立 MatMul**；diffusion_gemma 是一个融合 MatMul 再 chunk，不匹配。
- **2-GEMM pattern**（`moe_op_fusion.cpp:239`）虽是融合 gate_up，但写死 gpt-oss 式结构：
  `Slice(step=2 交错)→Clamp→Add→Minimum→Swish`；diffusion_gemma 是连续两半 chunk（step=1）、
  GEGLU、无 clamp/bias，也不匹配（且该 pattern 只接 `Swish`，不接 `Gelu`）。

→ **需要新增专家融合 pass**：`MatMul(gate_up) → 两个连续 Slice → Gelu(tanh)(gate) * up → MatMul(down)`，
产出 `GEMM3_SWIGLU` + `GEGLU_TANH`，融合 gate_up 权重布局 `[E, 2*inter, hidden]`。

### 缺口 2 Router 侧：`per_expert_scale` 的处理
diffusion_gemma 在归一化 Divide 之后多了 `Gather(per_expert_scale, idx) * weights`。
- 通用 `MOE` 的 routing-weights pattern（`fuse_moe_experts.cpp:247`）Divide 后没有这一步，会打断匹配。
- compressed GPU 路径**已有** SOFTMAX per-expert scale 的吸收（折进 down scale 的 WA），
  但要求 per_expert_scale 是 **Constant**（diffusion_gemma 的 learned 参数导出后即常量，✓）。

### 缺口 3 GPU kernel：融合 gate_up 的 GEGLU 连续-chunk 布局需确认
GEMM3 kernel 文档是 gate/up/down 三个独立权重；融合 gate_up 的连续切分目前由 GEMM2 kernel
（交错 step=2 + clamp）处理。需确认/补一条「连续 chunk + GEGLU」的切分分支（激活本身已具备，工作量小）。

### 不需要专门支持的部分（澄清）
- **并行 dense MLP（伪 shared expert）**：保持普通 MLP ops、在 MoE op 外做加法即可，无需融进 MoE op。
  GPU 的 `num_shared_expert` 路径语义不同，不要强行套用。
- **router 前的 RMSNorm × scale × hidden^-0.5**：在 router MatMul 上游，留作普通 ops，
  融合 pattern 用 `any_input()` 接 softmax，不受影响。

### 结论
支持 diffusion_gemma MoE 的**核心缺口是图变换层，而非算子/kernel 层**——
激活（GEGLU_TANH）和 3-GEMM 运行时都已具备。优先级：
1.【必须，主要】新增专家融合 pass（融合 gate_up + GEGLU + down）。
2.【必须】router 处理 `per_expert_scale`（已可复用 compressed 折叠）。
3.【需验证，可能小改】GPU kernel 对连续-chunk 融合 gate_up + GEGLU 的支持。
4.【验证】PyTorch frontend 能否产出可被上述 pattern 稳定匹配的 IR。

---

## 4. 已实施的改动：暴露 `fused_router` builder（路线 A）

### 背景与决策
- 代码里**没有**独立的 `fused_router` / `MoERouterFused` op；多处注释提到的
  "MoERouterFused node" 名不副实，实际产物是 `MOE3GemmFusedCompressed`，router 是其**内部融合行为**。
- 决策（已与用户确认）：
  - **暴露层级 = genai builder 函数（路线 A）**，不新增 OV op、不动 GPU kernel。
  - **per_expert_scale = 复用现有 WA 折叠进 w2(down) scale**。
- 思路：把原本内联在 `moe3gemm_fused_compressed()` 里的 router 子图抽成可复用的 `fused_router()`，
  图形状保持不变 → 现有 GPU 融合照常命中，diffusion_gemma 这类「router 不同、experts 相同」的模型
  可在 modeling 层直接组合 `fused_router(..., per_expert_scale)`。

### 改动文件（均在 `thirdparty/openvino.genai/`）

**1. `src/cpp/src/modeling/ops/ops.hpp`**
新增声明（位于 `moe3gemm_fused_compressed` 上方同段），带文档说明：
```cpp
std::pair<Tensor, Tensor> fused_router(const Tensor& input,
                                       const Tensor& gate_inp_weight,
                                       int32_t num_experts,
                                       int32_t top_k,
                                       const Tensor& per_expert_scale = Tensor());
```
返回 `{topk_weights, topk_indices}`。`per_expert_scale` 可选，**必须是 Constant** 才能被 GPU
`FuseMOE3GemmCompressed` 折叠；传默认构造的 `Tensor()` 表示不启用。

**2. `src/cpp/src/modeling/ops/ops.cpp`**
- 新增 `fused_router()`：抽出 router 子图
  `MatMul → Softmax(axis=1) → TopK(top_k) → ReduceSum → Divide`，
  并在 Divide 后接**可选**的 `Gather(per_expert_scale, topk_idx, axis=0) → Multiply`。
  Gather/Multiply 的形状刻意对齐 `fuse_moe_3gemm_compressed.cpp:64-67` 的
  `sm_per_expert_gather` / `sm_norm_scaled` optional 节点，使 GPU 融合的折叠分支能命中。
  另含 `gate_inp_weight` 专家维与 `num_experts` 的一致性校验。
- `moe3gemm_fused_compressed()` 改为调用 `fused_router(input, gate_inp_weight, num_experts, top_k)`，
  删除原内联 router；其余逻辑不变。**生成的图与改动前逐字节等价 → 现有 7 个调用点零回归**
  （qwen3_moe / qwen3_omni_moe / qwen3_tts / qwen3_next / qwen3_5 等）。
- 同步修正本文件中名不副实的注释（MoERouterFused → MOE3GemmFusedCompressed）。

**3. `src/cpp/src/gguf_utils/building_blocks.cpp`**
订正 L1164 注释：`FuseMoERouter` / `MoERouterFused`
→ `FuseMOE3GemmCompressed` / `MOE3GemmFusedCompressed`。

### diffusion_gemma 在 modeling 中的用法
```cpp
auto [weights, indices] =
    ops::fused_router(hidden, gate_inp_w, num_experts, top_k, per_expert_scale);
// weights / indices 再喂给专家计算（MOECompressed 或后续专家 builder）
```

### 关键实现注意点
1. GPU 折叠分支（`fuse_moe_3gemm_compressed.cpp:208`）强制 `per_expert_scale` 与 `w2_scale`
   均为 `Constant`，否则会在 L228 assert 失败。故 `fused_router` 仅支持**常量** per_expert_scale。
2. Gather 的 indices 用 `topk->output(1)`（未 Convert）；GPU pattern 的 `sm_convert_topk` 是
   `optional<Convert>`，接原始 i64 indices 也能匹配。
3. `moe3gemm_fused_compressed` 默认不传 per_expert_scale → 图与现状一致，无回归。

### 状态与待验证
- 已修复一处编译错误：args 中 `topk_weights->output(0)` 改为 `topk_weights`
  （重构后该变量已是 `ov::Output<ov::Node>`）。
- 头文件 include 充分（`opset13.hpp` 覆盖 v8::Softmax/v8::Gather/v11::TopK；
  `except.hpp` 覆盖 `OPENVINO_ASSERT`），无新增 include。
- **未在本环境完成**：`clang-format-18`（未安装）、完整 build、运行验证。
  建议在 dev_env 跑 format + 构建 + `ops_test.cpp` 的 MoE 用例。

---

## 5. 已实施的改动：专家侧支持 GEGLU + 串通 per_expert_scale

### 决策（已与用户确认）
- diffusion_gemma 的专家**通过 genai builder 直接构图**（与 qwen3_moe 一致），
  **gate 与 up 保持为独立权重**，构建标准 moe3gemm 图（`MOECompressed` → GPU 上
  `FuseMOE3GemmCompressed` → `MOE3GemmFusedCompressed`）。
- **不新增 OV op、不新增 GPU kernel、不新增 transformation pass**。

### 关键发现
- 原计划的「新增专家融合 MatcherPass」**并不需要**：本栈的 MoE 模型不走「eager 专家算子
  → pattern 折叠」，而是 builder 直接构造 `MOECompressed` op。
- `MOE` op 的 `activation_type` 已**端到端打通**：`MOE::Config` → `visit_attributes`
  （`moe.cpp:48`）→ 序列化 → GPU 融合 `get_config()` 透传（`fuse_moe_3gemm_compressed.cpp:163`）
  → primitive → kernel JIT（`GATE_ACT_GELU_TANH`）。GPU kernel **已实现 GEGLU_TANH / GEGLU_ERF**。
- `validate_and_infer_types`（`moe.cpp:34`）只对 `GEMM2_BIAS_SWIGLU_CLAMP` 限制 SWIGLU；
  **`GEMM3_SWIGLU + GEGLU_TANH` 合法**。
- 因此唯一功能缺口：`moe3gemm_fused_compressed` builder **写死了 SWIGLU 激活**。

### 改动（`thirdparty/openvino.genai/src/cpp/src/modeling/ops/`）
- **ops.hpp**：`#include <openvino/op/moe.hpp>`；给 `moe3gemm_fused_compressed` 末尾新增两个
  带默认值的参数：
  - `ov::op::internal::MOE::Activation_type activation_type = SWIGLU`
  - `const Tensor& per_expert_scale = Tensor()`
- **ops.cpp**：
  - 把 `activation_type` / `per_expert_scale` 传入实现；`per_expert_scale` 透传给
    `fused_router(...)`（在归一化后 Gather+Multiply）。
  - 设置 `config.activation_type = activation_type`（之前缺失，等价于硬编码 SWIGLU）。
- **零回归**：7 个现有调用点（qwen3_moe / qwen3_omni_moe / qwen3_tts / qwen3_next /
  qwen3_5 等）末位实参均为 `ov::element::f16)`，新参数走默认值，行为不变。

### diffusion_gemma 用法
```cpp
ops::moe3gemm_fused_compressed(
    flat_f32, gate_inp_w,
    gate_exps_w, gate_exps_s, gate_exps_zp,   // gate / up 独立权重
    up_exps_w,   up_exps_s,   up_exps_zp,
    down_exps_w, down_exps_s, down_exps_zp,
    hidden_size, inter_size, num_experts, top_k, group_size,
    ov::element::f16,
    ov::op::internal::MOE::Activation_type::GEGLU_TANH,  // gelu_pytorch_tanh
    per_expert_scale_const);                             // learned [num_experts] 常量
```

### 执行路径说明
- 融合 MoE 路径为 **GPU-only**（与现有 MoE 模型一致：`can_use_fused_path()` GPU 门控，
  CPU 走 `routed_fallback` 分解路径）。CPU 插件无 fused MOE op。
- diffusion_gemma 的目标即 GPU 融合路径，GEGLU_TANH 已具备 kernel 支持。

---

## 6. 已实施的改动：diffusion_gemma modeling 接入 fused GPU MoE（in-flight INT4 量化）

把 `DiffusionGemmaMoEBlock` 从 eager dense ops 切到融合压缩 GPU MoE 路径
（`moe3gemm_fused_compressed` + `fused_router`），并在 modeling 层做 in-flight INT4 量化。

### 决策（已与用户确认）
- **量化位置 = modeling builder 内 in-flight**（`DiffusionGemmaMoEBlock` 自定义 weight loader）。
- **精度 = INT4 非对称**（u4 weights + u4 zps + f16 scales）。
- **保留 eager fallback**，按 qwen3 `can_use_fused_path()` 门控。

### 关键发现（迫使量化成为强制项）
- 融合 MoE GPU kernel（`moe_3gemm_swiglu_mlp.cl`，`DEQUANT_4BIT`）**只支持低比特**，
  **没有 f16 权重路径** → 走 fused 路径**必须**先量化 experts（与 qwen3 走预量化 GGUF 等价）。
- diffusion_gemma 的 **router 输入 ≠ experts 输入**（router 自带 RMSNorm×scale×hidden^-0.5），
  而原 `moe3gemm_fused_compressed` 用同一个 `input` 同时喂 router 与 experts
  → 给 builder 增加可选的独立 `router_input` 参数（GPU 上 router 是独立融合子图，输入不同无碍）。
- `gate_up_proj` 融合权重 `[E, 2I, H]`，HF `chunk(2, -1)` 为**连续两半**
  → 在 loader 里拆成**独立 gate / up 量化权重**（gate = 前 I 行，up = 后 I 行）。

### 改动文件（均在 `thirdparty/openvino.genai/`）

**1. `src/cpp/src/modeling/ops/ops.hpp` + `ops.cpp`** —— builder 增加独立 router 输入
- `moe3gemm_fused_compressed` 末尾再加一个带默认值的参数 `const Tensor& router_input = Tensor()`。
- 实现里：`const Tensor& router_in = (router_input 非空) ? router_input : input;`
  传给 `fused_router(router_in, ...)`；experts 仍用 `input`（→ `hidden_f16`）。
- **零回归**：7 个现有调用点末位实参仍为 `ov::element::f16)` 或 `per_expert_scale`，新参数走默认值。

**2. `.../models/diffusion_gemma/modeling_diffusion_gemma_text.hpp`** —— `DiffusionGemmaMoEBlock`
- 新增 `size_t group_size_ = 0;`。
- 新增 9 个 in-flight 量化权重 member（gate / up / down × weights / scales / zps）。
- 新增 `bool can_use_fused_path() const;`、`Tensor forward_fused(...) const;`、
  `Tensor forward_eager(...) const;`。

**3. `.../models/diffusion_gemma/modeling_diffusion_gemma_text.cpp`**
- 匿名命名空间新增：
  - `quantize_moe_int4_asym_view(src, E, n, k, group_size, expert_stride, row_base)`：
    产出与 `quantize_q41`（`ov_ops_tests/cpp/moe_q41_moe3gemm_test.cpp`）逐字段一致的布局
    —— weights `u4 [E, out, in]`（沿 in 分组），scales `f16 [E, G, out]`，zps `u4 [E, G, out]`，
    dequant 语义 `(q - zp) * scale`；含 `[0,15]` clamp。
  - `set_u4`（偶 idx 低半字节 / 奇 idx 高半字节）、`pick_moe_group_size`
    （{128, 64, 32} 中首个能同时整除 hidden 与 inter 者，否则 0）。
- 构造函数：`group_size_ = pick_moe_group_size(hidden, inter);`；给 `experts.gate_up_proj`、
  `experts.down_proj` 装**自定义 weight loader**：
  - gate_up loader：读 `source.get_tensor(weight_name)` `[E, 2I, H]`，拆 gate（row_base = 0）/
    up（row_base = I），各量化为 INT4-asym 存入 member；`set_optional(true)` 且**不 bind f32 原权重**
    （释放最大的那批权重）。
  - down loader：`[E, H, I]` 量化为 INT4-asym。
  - `group_size_ == 0`（形状不可量化）时回退：直接 `bind` f32 权重走 eager。
- `forward`：
  - `forward(x)` → `forward(x, x)`。
  - `forward(router_input, experts_input)` 按 `can_use_fused_path()` 分派
    `forward_fused` / `forward_eager`（原 eager 实现整体重命名为 `forward_eager`）。
  - `forward_fused`：experts / router 各 reshape 到 `[T, H]` f32；**router 预处理在上游 eager 完成**
    （`rms_norm_no_scale × hidden^-0.5 ×（可选）router.scale`），`per_expert_scale` 取自
    `router.per_expert_scale`；调用 `moe3gemm_fused_compressed(..., GEGLU_TANH, per_expert, router_pre)`。

### 量化与门控不变量
- dequant `(q - zp) * scale`；scales/zps 以 `[E, G, out]` 传入，builder 的 `normalize_aux`
  会转置成 kernel 需要的 `[E, out, G]`。
- 沿**输入维**分组：gate / up 的 k = hidden，down 的 k = inter。
- `fused_router` 的路由 MatMul（`transpose_b = true`，`router.proj.weight [E, H]`）与 eager 的
  `ops::linear` 完全一致；router 与 experts 是仅在 topk_weights / indices 汇合的两个独立子图。
- `can_use_fused_path() = group_size_ > 0 && 三个量化权重 node 均存在`。

### 行为变化（需注意）
- 之前 diffusion_gemma MoE 恒走 eager；现在只要 expert 形状能被 group_size 整除即走
  `MOECompressed`（GPU 融合）。**CPU 用户也会进入 MOECompressed 路径**（之前是 eager）。
- 当前仓库无 diffusion_gemma 测试，故无 CPU 路径 CI 受影响。

### 状态与待验证
- `get_errors` 在改动的 hpp / cpp 与 `ops.hpp` / `ops_test.cpp` 上干净；`ops.cpp` 报的是
  IntelliSense 对 `OPENVINO_THROW` / `OPENVINO_ASSERT` 宏的**误报**（命中大量未改动的旧行，
  且同样的宏在 diffusion cpp 中无报错），非真实编译错误。
- **未在本环境完成**：`clang-format-18`（未安装）、完整 build、运行验证。
  建议在 dev_env 跑 format + 构建 + `ov_ops_tests/cpp/moe_q41_moe3gemm_test.cpp` 的 MoE 用例。

---

## 7. 后续工作（尚未做）

- 实测把 diffusion_gemma 导出为 OpenVINO IR / 直接构图后，dump MoE 子图核对 router 的
  Gather+Multiply 是否被 `FuseMOE3GemmCompressed` 的 `sm_per_expert_scale` 折叠分支命中
  （要求 per_expert_scale 为常量）。
- 端到端精度对齐（与 HF transformers 参考实现比对；注意 INT4-asym 量化引入的精度差异）。
- 在 dev_env 完成 `clang-format-18` + 完整 build + `moe_q41_moe3gemm_test.cpp` MoE 单测验证。
