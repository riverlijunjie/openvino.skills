# 3-bit LLM 压缩算法总结

> 目标：为 OpenVINO MoE GPU 推理路径添加 3-bit 权重支持，涵盖压缩（量化）与解压（dequant/推理 kernel）两个阶段。

---

## 一、算法全景

| 算法 | 类型 | 目标精度 | 压缩方式 | 解压复杂度 | 量化耗时（70B） |
|---|---|---|---|---|---|
| **BitNet b1.58** | QAT（训练级） | 1.58 bit | {-1,0,1} × 缩放因子 | 极低（乘法退化为加减） | 需从头训练 |
| **GPTQ / RTN** | PTQ（标量） | 2–4 bit | 逐组标量量化 | 极低 | <1 h |
| **SignRoundV2** | PTQ（标量 + 混合精度） | 2–3 bit | 符号梯度优化 rounding + 自适应层级 bit 分配 | 极低（标量 dequant） | ~2.5 h |
| **QuIP#** | PTQ（向量量化 VQ） | 2–4 bit | Hadamard 随机旋转 + E₈ lattice codebook（8D VQ） | 低（L1 cache 内查表） | ~270 h |
| **AQLM** | PTQ（加法向量量化 MCQ） | 2–3 bit | 加法量化（多码本叠加，8D × 2 codebook） | 中（1 MiB codebook，L1 不适配） | ~336 h |
| **QTIP** | PTQ（格码量化 TCQ） | 2–4 bit | Bitshift trellis + 计算型高斯码（1MAD / 3INST / HYB） | 中低（≤4 指令/weight，可并行） | 未公开 |
| **YAQA** | PTQ（TCQ + 更好 Hessian） | 2–4 bit | QTIP 框架 + Kronecker 因子化全局 Hessian 逼近 | 同 QTIP | 未公开 |
| **GGUF Q3_K** | PTQ（标量，k-quants） | 3.4375 bit | 超级块标量 INT3（2bit+1high bit），6-bit 分块 scale | 极低（shift+AND+sub+mul） | 分钟级 |
| **GGUF IQ3_XXS** | PTQ（4D D4 lattice VQ） | 3.0625 bit | 256-entry D4 lattice codebook，4D per index，7-bit 符号 | 低（4D LUT + sign flip） | 分钟级 |
| **GGUF IQ3_S** | PTQ（4D D4 lattice VQ） | 3.3125 bit | 512-entry D4 lattice codebook，9-bit index，8-bit 符号 | 低（4D LUT + sign flip） | 分钟级 |

---

## 二、各算法详细分析

### 1. BitNet b1.58（arXiv:2504.12285）— 训练级方法

**核心思路**：训练时将权重约束为 {-1, 0, +1}，精度等效 1.58 bit。激活保持 FP8 或 INT8。

**优点**：
- 解压最简单：W·x 退化为符号加减运算，无乘法
- 可定制专用硬件/内核，极致速度
- Microsoft 已开源 2B 模型（4T tokens 训练），性能对标同规模 FP16

**缺点**：
- **必须从头训练**，无法应用于已有 FP16/BF16 权重（无 PTQ 路径）
- 需要配套专用训练基础设施和大量数据
- 在现有 Transformer 推理栈中支持有限

**精度**：与 FP16 同规模模型相当（2B 模型验证）

**实现难度（OpenVINO PTQ 视角）**：❌ 不适用（非 PTQ，需改变训练流程）

---

### 2. GPTQ / RTN — 标量 PTQ 基准线

**核心思路**：逐层/逐组标量量化，GPTQ 用二阶 Hessian 信息优化 rounding。

**优点**：
- 工程最成熟，OpenVINO 已支持 GPTQ W4A16
- 极快量化（几分钟至数小时）
- 解压 kernel 简单

**缺点**：
- 2-bit 时精度严重下降（Llama2-7B W2 准确率仅 ~41.6%，远低于 FP16 64.7%）
- 3-bit 性能显著低于 AQLM / QTIP / QuIP#

---

### 3. SignRoundV2（arXiv:2512.04746）— Intel PTQ 方案 ✅ 推荐

**核心创新**：
1. **DeltaLoss 灵敏度度量**：梯度 × 量化偏差的一阶泰勒展开，比 Hessian 更准确
2. **逐层自适应 bit 分配**（动态规划）：将敏感层分配更高精度
3. **量化参数预搜索初始化**：基于 llama.cpp importance matrix 思路，改善 2-bit 收敛

**3-bit 实现方式**：混合精度 W2G128 + W4G128（平均 ≈3 bit）

**精度（W2A16，Llama 族）**：

| 模型 | FP16 | SignRoundV2 (2-bit) | SignRoundV2* (2-bit) | AQLM (2-bit 1x16) |
|---|---|---|---|---|
| Llama2-7B | 64.66% | 57.88% | 58.67% | 61.85% |
| Llama2-70B | 72.41% | 68.39% | 68.82% | 70.84% |
| Llama3-70B | 75.28% | 69.02% | 70.16% | 70.10% |

**混合精度 2.5-bit（W2+W4 混合）**：Llama2-70B 达 70.10%，接近 AQLM 2-bit 水平

**量化速度**：Llama2-70B 仅需 **2.5 小时**（vs AQLM 336h，QuIP# 270h）

**解压复杂度**：标量 dequant，与 INT4 完全一致，`group_size` 控制精度

**实现难度**：⭐⭐ 低（OpenVINO 已有 INT4 dequant 基础，扩展 INT2 + 混合精度即可）

**代码**：https://github.com/intel/auto-round

---

### 4. QuIP#（arXiv:2402.04396）— 向量量化 + 随机 Hadamard 旋转

**核心创新**：
1. **随机 Hadamard 变换（RHT）** 使权重分布变为近似 i.i.d 高斯，降低异常值影响
2. **E₈ lattice codebook**：8 维向量量化，高度对称，可 256× 压缩，恰好适配 L1 cache（~8KiB）
3. 支持 codebook fine-tuning

**3-bit 精度（Llama 2 7B，WikiText2 perplexity ↓）**：
- QuIP# 3-bit: 5.98（vs FP16 5.12，QTIP 3-bit: 5.85，AQLM 3-bit: 5.38）

**推理速度（RTX6000 Ada，Llama2-7B，batch=1）**：
- 2-bit: 186 tok/s；3-bit: 无官方数据（比 2-bit 略慢）
- AQLM 同配置：81.5 tok/s（慢 2× 以上）

**解压复杂度**：
- 需要逆 Hadamard 变换（O(n log n)）+ E₈ codebook 查表（L1 cache 内）
- 比标量 dequant 复杂，但 GPU 上高度并行

**实现难度**：⭐⭐⭐ 中（需 Hadamard 变换 kernel + E₈ lookup kernel）

**代码**：https://github.com/Cornell-RelaxML/quip-sharp

---

### 5. AQLM（arXiv:2401.06118）— 加法量化（MCQ）

**核心创新**：
- 将 ANNS（近似最近邻搜索）领域的 Additive Quantization 引入 LLM
- 每个权重向量表示为多个码本向量之和（默认 2 个码本 × 8 维 = 1x16 或 2x8）
- 跨 Transformer block 联合优化码本参数

**2-3-bit 精度**：Pareto 最优（在 <3 bit 区间），2-bit 最强

**3-bit（Llama 2 7B，WikiText2）**：perplexity = 5.38（最优）

**解压复杂度**：
- 码本大小 2^16 × 8 ≈ 1 MiB，**不适配 GPU L1 cache（通常 256KiB）**
- 查表存在 cache miss，实测推理速度仅约 81 tok/s（慢于 QuIP# 的 186 tok/s）
- 不支持完全并行解码

**量化耗时**：Llama2-70B 约 **336 小时**

**实现难度**：⭐⭐⭐⭐ 高（多码本加法查表 kernel，大 codebook 带宽瓶颈难以缓解）

**代码**：https://github.com/Vahe1994/AQLM

---

### 6. QTIP（arXiv:2406.11235）— 格码量化（TCQ）⭐ 质量+速度最佳

**核心创新**：
1. **Trellis-Coded Quantization（TCQ）**：将 Viterbi 算法应用于序列量化，线性时间实现超高维（>100）量化，突破 VQ 的指数缩放限制
2. **Bitshift trellis**：允许并行解码，无需存储 trellis 结构
3. **计算型高斯码（lookup-free）**：
   - `1MAD`：线性同余生成器（LCG）→ 近似高斯，仅 2 条 GPU 指令/weight
   - `3INST`：LCG + XOR + FP16 魔数，3 条指令/weight，精度更高
   - `HYB`：混合计算+查表，2KiB codebook 适配 L1 cache，可 fine-tune

**3-bit 精度（WikiText2 perplexity）**：

| 模型 | FP16 | QTIP 3-bit | QuIP# 3-bit | AQLM ~3-bit |
|---|---|---|---|---|
| Llama 2 7B | 5.12 | **5.85** | 5.98 | 5.38 (3.01b) |
| Llama 2 13B | 4.57 | **5.24** | 5.31 | 4.78 (3.03b) |
| Llama 2 70B | 3.12 | **3.26** | 3.35 | 3.36 (3.04b) |
| Llama 3.1 405B-Instruct | — | perp=2.05 | — | — |

**推理速度（RTX6000 Ada，Llama2-7B，batch=1）**：

| 精度 | QTIP | QuIP# | AQLM | FP16 |
|---|---|---|---|---|
| 2-bit | 188 tok/s | 186 tok/s | 81.5 tok/s | OOM |
| 3-bit | 161 tok/s | — | — | — |
| 4-bit | 140 tok/s | — | — | 55.9 tok/s |

**峰值内存带宽利用率**：>80%（接近理论峰值）

**解压复杂度**：
- ≤4 GPU 指令/weight（`1MAD`/`3INST`：2–3 指令，`HYB`：~2 指令）
- 支持并行解码（bitshift trellis 特性）
- 兼容 16×16 MMA tile，适配 Tensor Core

**量化流程**：在 QuIP# BlockLDLQ 框架内使用 TCQ，需要生成 Hessian（数小时 GPU 时间）

**实现难度**：⭐⭐⭐ 中（TCQ 核心逻辑清晰，LCG 计算码简单，但 Viterbi + BlockLDLQ 集成需要仔细实现）

**代码**：https://github.com/Cornell-RelaxML/qtip

---

### 7. YAQA（arXiv:2505.22988）— QTIP + 全局 Hessian 逼近

**核心创新**：
- 在 QTIP 基础上改用 **Kronecker 因子化** 对全模型 KL 散度的 Hessian 逼近（而非逐层 proxy Hessian）
- 比 LDLQ/GPTQ 将 KL 散度减少 **1/3**
- 在广泛模型和量化精度上达到 SOTA

**与 QTIP 关系**：代码基于 QTIP，解压 kernel 相同，主要差异在量化（压缩）阶段

**实现难度**：⭐⭐⭐ 中（解压同 QTIP；量化时需额外实现全局 Hessian 估计）

---

## 三、综合对比分析

### 精度排名（3-bit，PTQ，Llama2-7B WikiText2 perplexity，↓ 更好）

```
AQLM(3.01b): 5.38 > QTIP(3b): 5.85 > QuIP#(3b): 5.98 >> SignRoundV2(3b mixed): ~6.x
```

> 注：AQLM 的 3-bit 精度最优，但推理速度最慢

### 推理速度排名（3-bit，batch=1，7B 模型）

```
QTIP: 161 tok/s > QuIP#: ~160+ tok/s >> SignRoundV2(标量): GPU 利用率高但取决于实现 >> AQLM: ~70–80 tok/s
```

### 实现复杂度排名（对 OpenVINO GPU kernel 开发）

```
SignRoundV2 < QTIP (1MAD/3INST) < QuIP# (E8 lookup) < AQLM (大codebook) << BitNet(需训练)
```

---

## 四、OpenVINO 3-bit MoE 实施建议

### 推荐路径（优先级顺序）

#### 路径 A：混合精度标量量化（最快落地）✅
- **方案**：SignRoundV2 W2A16 + W4A16 混合，平均 ~3 bit
- **量化工具**：Intel auto-round（OpenVINO 生态兼容）
- **推理 kernel**：复用现有 INT4 dequant，新增 INT2 dequant 路径，按 group_size 混合
- **精度**：70B 模型约 ~94% FP16 精度恢复
- **开发量**：小（2–3 周）
- **适合 MoE**：是，每个 expert 权重独立量化，group 级 dequant 并行度好

#### 路径 B：QTIP TCQ 计算型解压（质量更高）🔧
- **方案**：QTIP 3-bit，1MAD 或 3INST 码
- **推理 kernel**：LCG 生成伪高斯 → bitshift trellis 并行解码 → 逆 Hadamard
- **精度**：比 SignRoundV2 好约 0.1–0.3 PPL（7B），速度接近理论峰值
- **开发量**：中（4–6 周，主要在 CUDA kernel + Hadamard）
- **适合 MoE**：是，MoE expert 矩阵独立，trellis 解码可按行并行

#### 路径 C：QuIP# E₈ codebook（成熟社区，质量中等）
- **方案**：QuIP# 3-bit E₈ VQ
- **推理 kernel**：Hadamard 逆变换 + E₈ 查表（8KiB）
- **精度**：略低于 QTIP，快于 AQLM
- **开发量**：中（E₈ codebook 特殊结构需专门优化）

#### 不推荐 / 低优先级
- **AQLM 3-bit**：精度最优，但解压 bandwidth-bound，适合 CPU 但 GPU 上性能不如 QTIP
- **BitNet**：非 PTQ，无法应用于现有权重

### MoE 特殊考虑

- MoE 的 expert 权重在推理时**稀疏激活**，单次推理只激活少数 expert（通常 2/N）
- 3-bit 压缩对 MoE 尤为重要：N×expert 参数量巨大，压缩比直接影响能否将 expert 常驻 GPU HBM
- `GEMM` vs `GEMV`：decode 阶段 batch=1 是 memory-bound GEMV，dequant 开销占主导；prefill 阶段是 compute-bound GEMM，dequant 可融合到 GEMM kernel
- 推荐对每个 expert 独立量化（per-expert group），允许不同 expert 使用不同精度（敏感 expert 保留 4-bit）

---

## 五、参考资料

| 论文/资源 | 链接 |
|---|---|
| BitNet b1.58 2B4T Technical Report | https://arxiv.org/abs/2504.12285 |
| QuIP#: Hadamard Incoherence + E₈ Lattice | https://arxiv.org/abs/2402.04396 |
| AQLM: Extreme Compression via Additive Quantization | https://arxiv.org/abs/2401.06118 |
| SignRoundV2: Low-Bit PTQ (Intel) | https://arxiv.org/abs/2512.04746 |
| QTIP: Trellis Coded Quantization | https://arxiv.org/abs/2406.11235 |
| YAQA: Model-Preserving Adaptive Rounding | https://arxiv.org/abs/2505.22988 |
| auto-round (Intel, SignRound 实现) | https://github.com/intel/auto-round |
| QTIP 代码 | https://github.com/Cornell-RelaxML/qtip |
| YAQA 代码 | https://github.com/Cornell-RelaxML/yaqa-quantization |

---

## 六、GGUF 3-bit 量化格式深度分析

> 参考：[unsloth/Qwen3.6-35B-A3B-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/tree/main)，llama.cpp `ggml-quants.c`

### GGUF 量化体系概览

GGUF（GPT-Generated Unified Format）是 llama.cpp/ggml 采用的二进制模型格式，内置多种量化类型。3-bit 相关格式分为两族：

| 族 | 代表类型 | 特点 |
|---|---|---|
| **k-quants**（标量量化） | Q3_K_S/M/XL | 超级块（super-block）内标量量化，纯标量解压，不依赖 imatrix |
| **IQ-quants**（格码量化） | IQ3_XXS, IQ3_S | D4/D6 lattice codebook，向量解压，依赖 importance matrix |

---

### 1. Q3_K（k-quants 3-bit）— `GGML_TYPE_Q3_K`

**物理布局（一个 super-block = 256 weights）**：

```c
typedef struct {
    uint8_t  hmask[QK_K/8];   // 32 bytes: 每个 weight 的第 2 bit（高位）
    uint8_t  qs[QK_K/4];      // 64 bytes: 每个 weight 的低 2 bit（4 weights/byte）
    uint8_t  scales[12];      // 12 bytes: 16 个 sub-block 的 6-bit scale（16×6=96 bits）
    ggml_fp16_t d;            // 2 bytes: super-block FP16 scale
} block_q3_K;
// 总计：110 bytes / 256 weights = 3.4375 bpw
```

**编码规则**：
- 量化范围：有符号整数 `[-4, 3]`（3-bit signed）
- 每个 weight 的整数值 = `qs[j] & 3` (2 low bits) `| (hmask[m] & hm ? 4 : 0)` (1 high bit) - 4
- sub-block scale（6-bit）相对于 super-block scale，表示范围 `[-32, 31]`

**解压公式**：
```
w = d_super × sub_scale × int3_value
其中 sub_scale = 6bit_val - 32（可为负数，支持正负 sub-block）
```

**Q3_K_S / Q3_K_M / Q3_K_XL 区别**：
- 这些后缀由量化工具（llama.cpp `llama-quantize`）决定哪些层用 Q3_K，哪些用更高精度：
  - `Q3_K_S`（Small）：大部分层 Q3_K，少数高敏感层保留 Q4_K
  - `Q3_K_M`（Medium）：大部分层 Q3_K，attention 层 Q4_K，输出层 Q5_K
  - `Q3_K_XL`（Unsloth 扩展）：更多层保留高精度，混合更激进
- 对 Qwen3.6-35B-A3B（MoE 模型），文件大小差异（15.4→16.8 GB）反映了不同层保留精度的比例

**精度**：
- Llama 2 7B WikiText2 perplexity：Q3_K_M ≈ 6.46（vs FP16 5.12，Q4_K_M 4.85）
- 精度明显低于 IQ3_XXS / IQ3_S

**解压复杂度**：⭐ 极低
- 纯标量 dequant：`int3 → sub_scale → FP16`，3 次乘法
- 无查表，无变换
- GPU kernel 实现最简单

**实现难度（OpenVINO）**：⭐ 很低（有 INT4 基础，扩展到 3-bit 存储格式即可）

---

### 2. IQ3_XXS（3.0625 bpw）— `GGML_TYPE_IQ3_XXS`

**物理布局（一个 super-block = 256 weights）**：

```
block_iq3_xxs:
  qs[QK_K/4 + QK_K/32*4] = qs[96]  // 96 bytes：grid indices + scales_and_signs
  d: FP16  // 2 bytes：super-block scale
总计：98 bytes / 256 weights = 3.0625 bpw
```

详细布局（每 32-weight 子块）：
- **8 字节 grid indices**：8 个 8-bit 索引，每个索引指向 256-entry D4 lattice codebook（`iq3xxs_grid`），每个 grid entry 解码 4 个 odd integers `{1,3,5,7,9,...}`
- **4 字节 scales\_and\_signs**：高 4 bits = 子块 scale（共 16 级），低 28 bits = 7-bit sign mask × 4 组（每组 8 个 weights 的符号位）

**解压公式（per 32-weight block）**：
```python
db = d * (0.5 + (aux32 >> 28)) * 0.5  # sub-block scale
for each 8-weight group:
    grid_vals = iq3xxs_grid[index]  # 4 values from codebook
    signs = ksigns_iq2xs[(aux32 >> 7*l) & 127]  # 7 sign bits → 8 signs
    w[j] = db * grid_vals[j] * sign[j]
```

**Codebook 特性**：
- 256-entry D4 lattice，每 entry 含 4 个 odd 整数
- codebook 仅 256×4 = 1KB，完全适配 GPU L1 cache
- 与 IQ2_XXS 共享相同 codebook 设计思路，但扩展到 3-bit 密度

**量化要求**：**必须使用 importance matrix（imatrix）**
- 权重重要性矩阵用于优化 scale 分配，显著改善 perplexity
- 无 imatrix 时精度接近甚至劣于 Q3_K
- unsloth 仓库提供专用 `imatrix_unsloth.gguf_file`（192 MB）

**精度（Llama 2 7B WikiText2）**：IQ3_XXS ≈ 6.09（优于 Q3_K_M 6.46，略差于 IQ3_S）

**解压复杂度**：⭐⭐ 低中
- 需要 codebook lookup（1KB，L1 cache 友好）
- 需要符号位重建（bitwise op）
- 比 Q3_K 多一层 table lookup，但 codebook 很小

**实现难度（OpenVINO）**：⭐⭐ 中低（需实现 256-entry codebook lookup kernel，sign bit 解码）

---

### 3. IQ3_S（3.3125 bpw）— `GGML_TYPE_IQ3_S`

**物理布局（一个 super-block = 256 weights）**：

```
block_iq3_s（总计约 106 bytes / 256 weights = 3.3125 bpw）:
  qs[QK_K*3/8]  // 96 bytes：8-bit low 部分的 grid indices
  qh[QK_K/32]   // 8 bytes：每个 index 的第 9 bit（高位）
  signs[QK_K/8] // 32 bytes：sign masks
  scales[QK_K/64] // 4 bytes：4-bit scale per 64-weight block
  d: FP16       // 2 bytes
```

实际 sizeof = (96 + 8 + 32 + 4 + 2) = ... 需校正，实测约 106 bytes。

**与 IQ3_XXS 的区别**：
- codebook 扩展为 **512 entries**（9-bit 索引，vs IQ3_XXS 的 8-bit/256 entries）
- 精度更高（更细粒度的格点覆盖）
- sign bits 存储为独立字段（`signs[]`），而非打包在 scales_and_signs 中
- 4-bit per sub-block scale（而非 IQ3_XXS 的 4-bit per 32-weight block）

**解压公式**：
```python
db = d * (1 + 2*(scales[ib/2] >> (4*(ib%2))))  # 4-bit scale decoded as odd integers
grid_index = qs[k] | (qh_high_bit)  # 9-bit index → 512-entry codebook
w[j] = db * grid[j] * sign  # grid entry: 4 values of odd integers
```

**精度（Llama 2 7B WikiText2）**：IQ3_S ≈ 5.87（优于 Q3_K_M 6.46，接近 QTIP 3-bit 5.85）

**解压复杂度**：⭐⭐ 低中（同 IQ3_XXS，codebook 扩展至 512×4 = 2KB，仍适配 L1 cache）

**实现难度（OpenVINO）**：⭐⭐ 中（与 IQ3_XXS 相似，codebook 略大，增加 9-bit 索引重建）

---

### 4. MXFP4_MOE — MoE 专用 MicroScaling FP4

```
文件：Qwen3.6-35B-A3B-MXFP4_MOE.gguf（21.7 GB）
```

**格式定义**：
- 非标准 3-bit，而是 **4.25 bpw** 的 MicroScaling FP4（OCP MX 规范）
- Data type：E2M1（2 exponent bits + 1 mantissa bit + 1 sign = 4-bit）
- Scale type：E8M0（8-bit block exponent，per 32 weights）
- 实际 bpw = 4 + 8/32 = **4.25 bpw**

**MoE 特殊性**：
- 仅对 MoE expert 权重（`experts.{gate,up,down}_proj`）应用 MXFP4
- 其他层（attention、layer norm 等）保持较高精度
- 意图：expert 权重占总参数的 ~85%，MXFP4 压缩可大幅减少 HBM 占用
- 21.7 GB vs Q3_K_M 16.6 GB：MXFP4 精度高但压缩率低于 3-bit

**硬件支持**：NVIDIA Ada Lovelace（H100/4090 支持 MX 格式 GEMM）；需专用 CUDA kernel

**实现难度（OpenVINO GPU）**：⭐⭐⭐ 中（OpenVINO 已在 GPU 插件中实验性支持 MXFP4）

---

### 5. GGUF 3-bit 格式横向比较

| 格式 | bpw | Codebook | imatrix | 解压复杂度 | L2 7B PPL | 实现难度 |
|---|---|---|---|---|---|---|
| **Q3_K** | 3.4375 | 无（标量） | 不需要 | 极低（3×乘法） | ~6.46 | ⭐ |
| **IQ3_XXS** | 3.0625 | 256-entry D4 (1KB) | **必须** | 低（查表+符号） | ~6.09 | ⭐⭐ |
| **IQ3_S** | 3.3125 | 512-entry D4 (2KB) | **必须** | 低（查表+符号） | ~5.87 | ⭐⭐ |
| **MXFP4_MOE** | 4.25 | 无（FP4 E2M1） | 不需要 | 低（FP4 decode） | N/A | ⭐⭐⭐ |

**与学术方法精度对比（Llama 2 7B WikiText2 perplexity↓）**：

```
AQLM(3.01b): 5.38 < QTIP(3b): 5.85 ≈ IQ3_S(3.31b): 5.87 < IQ3_XXS(3.06b): 6.09 < Q3_K_M(3.44b): 6.46
```

- **IQ3_S 精度接近 QTIP 3-bit**，且实现更简单（无需 Hadamard + trellis）
- IQ3 系列的核心优势：codebook 适配 L1 cache（1–2 KB），解压延迟低
- AQLM 在 <3-bit 区间最优，但 codebook 1 MiB 造成 GPU 推理带宽瓶颈

---

### 6. GGUF 3-bit 在 OpenVINO MoE GPU 路径的适用性评估

#### 可行性分析

| 因素 | Q3_K | IQ3_XXS / IQ3_S |
|---|---|---|
| **权重加载** | 标准 GGUF tensor，直接读取 | 同上，格式公开 |
| **GPU dequant kernel** | 只需 bit 拆分 + 标量乘法 | 需实现 codebook lookup（1–2 KB per thread block） |
| **MoE 兼容性** | ✅ 每 expert 独立量化，group 粒度好 | ✅ 同上，sub-block 32 weights 与 expert 矩阵行对齐 |
| **imatrix 依赖** | ❌ 不需要 | ⚠️ 量化阶段必须（推理时不需要） |
| **现有 OV 基础** | 有 INT4 dequant kernel 可复用 | 需新增 codebook lookup kernel |
| **标准化程度** | GGUF 规范，社区广泛使用 | GGUF 规范，但比 Q-quants 复杂 |

#### 推荐策略（GGUF 路径）

1. **首选支持 IQ3_S** 作为 GGUF 3-bit 输入格式：
   - 精度最接近学术 SOTA（IQ3_S ≈ QTIP 3-bit）
   - 2KB codebook 适配 GPU L1 cache，解压开销可接受
   - 比 AQLM 简单得多（无大 codebook 问题）

2. **同时支持 Q3_K** 作为快速 fallback：
   - 无 imatrix 依赖，任意权重可量化
   - 解压最简单，可共享 INT4 kernel 基础设施

3. **MXFP4_MOE 单独路径**：
   - 目前 4.25 bpw，不属于严格 3-bit 范畴
   - 但 Unsloth 的 MXFP4_MOE 是社区热门格式，可考虑单独支持

4. **IQ3_XXS** 低优先级：
   - bpw 最低但精度劣于 IQ3_S
   - codebook 更小（1KB），实现比 IQ3_S 简单
   - 若需要最小文件尺寸时考虑

#### 与内部路径（路径 A/B）的关系

- GGUF 3-bit 是**推理输入格式**，SignRoundV2 / QTIP 是**量化压缩方法**
- 两者正交：可以用 SignRoundV2 量化后导出为 GGUF Q3_K 格式，或直接加载社区 GGUF 文件
- 对于 MoE 推理，支持 GGUF IQ3_S 加载路径，可直接使用 Unsloth 等社区预量化模型，无需自行量化

---

### 7. llama.cpp GGUF 3-bit 相关技术资源

| 资源 | 链接 |
|---|---|
| ggml-quants.c（Q3_K / IQ3 实现） | https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-quants.c |
| ggml-common.h（block 结构体定义） | https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-common.h |
| GGUF 格式规范 | https://github.com/ggml-org/ggml/blob/master/docs/gguf.md |
| Unsloth Qwen3.6-35B-A3B GGUF | https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/tree/main |
| k-quants 原始 PR（#1684） | https://github.com/ggml-org/llama.cpp/pull/1684 |
| IQ3 quantization PR | https://github.com/ggml-org/llama.cpp/pull/5765 |

---

## 七、各 3-bit 量化方法的权重 Layout 设计

> 针对 GPU GEMV（decode, batch=1）和 GEMM（prefill, batch>1）两种推理路径，分析每种 3-bit 量化方法的最优权重内存布局。

### GPU Layout 基础约束

在讨论具体方法前，确立三条 GPU 硬件约束：

| 约束 | 说明 | 影响 |
|---|---|---|
| **Coalesced Access** | 同一 warp（32 threads）访问应落在连续的 128-byte cache line | 决定 K 方向（input 维度）的打包顺序 |
| **Tensor Core Tile** | WMMA/HMMA 对 B 矩阵（权重侧）有 tile 格式要求（16×16 或 32×32） | GEMM 时权重需要 interleaved 或 column-major 布局 |
| **Dequant 粒度对齐** | 解压单元（block/group）粒度应能被 warp 整除 | 决定 block size 是否需要 padding |

**GEMV vs GEMM 访问模式差异**：

```
GEMV（decode, batch=1）：
  - 输出 y[m] = Σ_k  W[m][k] * x[k]
  - 每个输出 y[m] 读取权重矩阵的一行（K 个 weights）
  - → 权重 Row-major 布局（同一行 K 维连续），warp 协作读一行

GEMM（prefill, batch>1）：
  - 输出 Y[m][b] = Σ_k  W[m][k] * X[k][b]
  - 按 tile 读取 W 的 K×N tile（N = 输出维度）
  - → 权重 Column-major 或 tile-interleaved 布局，warp 读取跨多行的同 K 位置
```

---

### 1. Q3_K — 标量 3-bit，super-block 256 weights

**Block 内部结构**（110 bytes，256 weights）：

```
block_q3_K 内存布局：
偏移  大小   字段         内容
  0   32B   hmask[32]    第 2 bit（高位）：bit[2] of weight[0..255]，每 bit 一个 weight
 32   64B   qs[64]       低 2 bit：bit[0:1] of weight[0..255]，4 weights/byte，按 128-weight 段 interleaved
 96   12B   scales[12]   16 个子块的 6-bit scale，每两个子块 pair 打包进 3 字节
108    2B   d            FP16 super-block scale
```

**权重矩阵存储（形状 M×K，行主序）**：

```
Row 0:  [blk(0,0)] [blk(0,1)] ... [blk(0,K/256-1)]   ← K/256 个 block_q3_K
Row 1:  [blk(1,0)] [blk(1,1)] ...
...
Row M-1:[blk(M-1,0)] ...

每行占用：(K/256) × 110 bytes
推荐行首地址对齐：128 bytes（GPU cache line）
```

**GEMV 最优分配（Decode）**：

```
Thread block：处理 TILE_M 行（输出），每行一个 warp
Warp（32 threads）：处理一行的所有 blocks，每次处理 1 个 block_q3_K：

  步骤 1：thread 0 加载 d（2B），warp 广播
  步骤 2：thread 0–2 加载 scales[12]（12B），warp shuffle 分发 16 个 sub-scale
  步骤 3：所有 thread 协作加载 hmask[32]（32B，每 thread 1 byte）
  步骤 4：所有 thread 协作加载 qs[64]（64B，每 thread 2 bytes → 8 weights）
  步骤 5：各 thread 重建 8 个 int3 值，乘以对应 sub-scale × d，累加到寄存器

每个 warp 处理 1 个 block_q3_K = 256 weights，需 5 步加载
```

**GEMM 最优布局（Prefill）**：

Q3_K 的三段式结构（`hmask`/`qs`/`scales`）不适合直接用于 Tensor Core；推荐 **dequant-to-INT4** 预处理：

```
预处理方案：将 block_q3_K 展开为 INT4 格式
int3 ∈ [-4,3] → int4 = int3 + 4 ∈ [0,7]（仅用低 3 bit，高 1 bit 恒为 0）

展开后 INT4 layout（CUTLASS 兼容）：
  qweight_int4[K//8][M]，每 int32 存 8 个 INT4，沿 K 方向打包
  scales_fp16[K//256][M]，保留 super-block scale
  sub_scales_fp16[K//16][M]，保留 sub-block scale（提前 decode 为 FP16）

GEMM kernel 使用 INT4 × FP16 混合精度 GEMM，scale 融入 epilogue
```

**MoE Expert 打包**：

```
Expert e 的权重：连续存放 M_e × (K/256) 个 block_q3_K
  size = M_e * ceil(K/256) * 110 bytes

Expert 边界对齐：256 bytes（2 个 cache line），避免跨 expert 的 cache 污染
Expert 指针表：uint64_t expert_offsets[n_experts]，指向各 expert 起始位置
```

---

### 2. IQ3_XXS — D4 lattice 256-entry codebook，3.0625 bpw

**Block 内部结构**（98 bytes，256 weights，8 个 32-weight 子块）：

```
block_iq3_xxs 内存布局：
偏移  大小   字段
  0   96B   qs[96]
             子块 0–7 交织存放：
             [blk0: 8B indices + 4B scales_signs] × 8
             具体：子块 k 的 indices 从 qs[12*k] 开始（8 bytes）
                  子块 k 的 scales_and_signs 从 qs[12*k+8] 开始（4 bytes = uint32）
 96    2B   d（FP16 super-block scale）

子块内 indices 对应关系：
  qs[12*k+0..7]  → 8 个 8-bit grid index，每个查 iq3xxs_grid[idx]（4 values）
  qs[12*k+8..11] → 1 个 uint32：
                    bits[31:28] = 4-bit scale（sub-block 幅度，共 16 级）
                    bits[27:21] = 7-bit sign mask（第 4 组 8 weights 的符号）
                    bits[20:14] = 7-bit sign mask（第 3 组）
                    bits[13:7]  = 7-bit sign mask（第 2 组）
                    bits[6:0]   = 7-bit sign mask（第 1 组）
```

**Codebook 常驻布局**：

```c
// IQ3_XXS codebook：256 × 4 个 int8（odd integers，1,3,5,...15）
// 放入 GPU constant memory（对所有 SM 广播），大小 = 1 KB
__constant__ int8_t iq3xxs_grid[256][4];  // 1 KB

// sign lookup：128 entries，每 entry 8 个 {-1,+1} packed 为 uint8
__constant__ uint8_t ksigns_iq2xs[128];   // 128 B

// 两者合计 ~1.1 KB，完整适配 L1 constant cache（通常 48 KB）
```

**GEMV 最优分配（Decode）**：

```
每个 warp 处理 1 个 32-weight 子块：

  Thread 0–7：各加载 1 个 index（1 byte），查 iq3xxs_grid
              → 得到 4 个 odd int（4 bytes），共 32 个 values
  Thread 0：加载 scales_and_signs（4 bytes = uint32），warp 广播
  所有 Thread：decode sign bit，multiply-add
              Thread i → weight[i] = db × grid[i%4] × sign[i]
  warp_reduce → 累加到输出

处理一个 block_iq3_xxs（8 子块）需 8 次迭代
→ 吞吐：256 weights / warp / 8 iter = 32 weights/iter（效率高）
```

**GEMV 推荐矩阵布局（行主序，原始格式）**：

```
Row 0:  [blk_0] [blk_1] ... [blk_{K/256-1}]
Row 1:  [blk_0] ...

行首 128-byte 对齐；每个 block 98 bytes，相邻 block 间无 padding
（98 byte block 不能整除 128，一个 cache line 可跨两个 block）
```

**GEMM 最优布局（Prefill）**：

对于 prefill，建议将 indices 从子块内布局重整为 **K-major 跨行连续**：

```
重整后（Column-major Interleaved）：
  indices[K/8][M]    → 每一行 M 个 row 在同一 K 位置的 8 indices，连续存放
  signs[K/32][M]     → 每一行 M 个 row 在同一子块的 sign，连续存放
  sub_scales[K/32][M]→ sub-block scale，连续存放
  d[1][M]            → super-block scale

好处：warp 沿 M 方向加载 32 个 row 的同一 K 位置 index，实现 coalesced access
      codebook lookup 结果在 warp 内无 bank conflict（不同 thread 访问不同 entry）
```

**MoE Expert 打包**：

```
Expert e：M_e 行，每行 K/256 个 block_iq3_xxs
  连续存放，expert 边界 128-byte 对齐
  codebook 共享（不随 expert 复制），放在 constant memory
```

---

### 3. IQ3_S — D4 lattice 512-entry codebook，3.3125 bpw

**Block 内部结构**（约 106 bytes，256 weights）：

```
block_iq3_s 内存布局：
偏移  大小   字段
  0   96B   qs[96]      8 个 32-weight 子块 × 8 个 8-bit index（9-bit 的低 8 位）
 96    8B   qh[8]       每子块 1 字节：8 个 index 的第 9 bit（高位）
             qh[k] bit i = index[8*k+i] >> 8
104   32B   signs[32]   每子块 4 字节 sign mask（8 signs/byte × 4 bytes = 32 signs）
136    4B   scales[4]   每 64-weight 组 1 个 4-bit scale，2 个 scales/byte → 4 bytes 存 8 个
140    2B   d           FP16 super-block scale
总计：142 bytes / 256 weights（需确认实际 sizeof）

注：实际 bpw = 142 × 8 / 256 = 4.44 bpw（含 scale overhead）
```

**IQ3_S vs IQ3_XXS 关键 Layout 差异**：

| 差异点 | IQ3_XXS | IQ3_S |
|---|---|---|
| Codebook 大小 | 256-entry，8-bit index | 512-entry，**9-bit index** |
| Index 高位 | 内嵌在 scales_and_signs | 独立 `qh[]` 字段 |
| Sign 存储 | 打包在 scales_and_signs | 独立 `signs[]` 字段 |
| Scale 粒度 | 32-weight/sub-block（4-bit） | 64-weight/sub-block（4-bit） |
| Codebook 内存 | 1 KB | **2 KB** |

**9-bit 索引重建**（关键 decode 步骤）：

```c
uint16_t idx = (uint16_t)qs[8*k + i]            // 低 8 bit
             | (((qh[k] >> i) & 1) << 8);       // 高 1 bit
const int8_t* grid = iq3s_grid[idx];            // 512-entry lookup
```

**GEMV 推荐分配（Decode）**：

```
每 warp 处理 1 个 32-weight 子块（对应 qs 中 8 bytes + qh 1 byte + signs 4 bytes）：

  Thread 0–7：加载 qs[8*k..8*k+7]（8 bytes，1 byte/thread）
  Thread 8：  加载 qh[k]（1 byte），warp 广播
  Thread 0–7：重建 9-bit index → 查 iq3s_grid（512-entry，2KB）
              得到 4 个 odd int values
  Thread 0：  加载 signs[4*k..4*k+3]（4 bytes = 32 signs），warp 广播
  Thread 0：  decode sub-scale from scales[k/2]（4-bit），warp 广播
  所有 Thread：multiply-add：w[i] = db × grid_val × sign
```

**GEMM 推荐布局（Prefill）**：

与 IQ3_XXS 相同原则，将 qs/qh/signs 字段分离并按 K-major 排列：

```
qs_kmajor[K/8][M]      → 8-bit low index，按 K 方向各行连续
qh_kmajor[K/256][M]    → 9-bit 高位 byte，按子块 K 位置各行连续
signs_kmajor[K/32][M]  → sign bytes，按子块各行连续
scales_fp16[K/64][M]   → 预展开为 FP16（避免 decode 时的 4-bit 计算）
d_fp16[1][M]           → super-block scale
```

---

### 4. SignRoundV2（INT2 + INT4 混合精度）— 标量量化

**基本单元**（group_size = 128，INT2）：

```
INT2 pack：128 weights → 128×2/8 = 32 bytes（qweight，int32 × 8，每 int32 存 16 个 INT2）
scale：    1 × FP16（2 bytes）
zp：       可选，1 × INT4 或 FP16

INT4 pack：128 weights → 128×4/8 = 64 bytes（qweight，int32 × 16）
scale：    1 × FP16
zp：       可选
```

**GPTQ-compatible 矩阵布局（形状 K×N，沿 K 打包）**：

```
// INT4 weight matrix (GPTQ 原始格式，沿 K 方向打包)：
qweight_int4: shape = (K//8, N), dtype = int32
  qweight_int4[k//8][n] = pack8(W[k][n], W[k+1][n], ..., W[k+7][n])
  即 8 个沿 K 方向的权重打包进一个 int32，N 方向不打包

// INT2 扩展：
qweight_int2: shape = (K//16, N), dtype = int32
  qweight_int2[k//16][n] = pack16(W[k..k+15][n])

// scales：
scales: shape = (K//group, N), dtype = float16

// 混合精度（同一层内 INT2 + INT4 按 group 交替）：
// group [0..Gi-1]   → INT2（低精度 group）
// group [Gi..Gtot-1]→ INT4（高精度 group）
// 或：按行（通道）混合，敏感通道 INT4，其余 INT2
// 统一用 qweight + precision_mask 数组标记每个 group 的精度
precision_mask: shape = (K//group, N), dtype = uint8  // 0=INT2, 1=INT4
```

**GEMV 最优分配（Decode，列主序视角）**：

```
对 GEMV y = W × x，W 形状 (M, K)：
  - 每个 thread 负责计算 1 个输出 y[m]
  - 需读取 W 的第 m 行（所有 K 个权重）
  - 在 qweight(K//8, M) 布局中，一行的数据分散在所有 qweight 行中

→ 推荐 thread 访问方向：
  thread m → 顺序读取 qweight[:][m]（N 维固定，遍历 K 维）
  K//8 个 int32，每次加载 4 bytes，decode → 8 个 INT4 → 乘 x → 累加

→ 对应内存需求：
  列主序（column-major）会使相邻 thread 访问相邻 column，造成 stride access
  推荐保持 GPTQ 格式（行主序，K 方向打包）：
    row m 的数据：qweight[0][m], qweight[1][m], ... （每隔 N*4 bytes）
  为避免 stride，改用 Transposed 格式：
    qweight_T: shape = (N, K//8), dtype = int32（列主序变为行主序）
    qweight_T[n][k//8] = pack8(W[k..k+7][n])
  → thread m 读取 qweight_T[m][:]，连续内存，coalesced access
```

**GEMM 最优布局（Prefill，CUTLASS INT4 Interleaved）**：

```
CUTLASS 推荐 INT4 B-matrix layout（B = weight matrix, K×N）：
  interleaved_int4: shape = (K//64, N, 64), dtype = int4×64
  即 沿 K 方向每 64 个为一组，同一 K-group 的 N 方向连续
  → thread block 可加载完整 16×64 INT4 tile 用于 WMMA

对 INT2，类似处理（K 方向每 128 个为一组）：
  interleaved_int2: shape = (K//128, N, 128), dtype = int2×128

混合精度：
  将 INT2 group 填充（zero-pad）为 INT4 存储（2-bit 值存在低 2 位），
  用单个 INT4 GEMM kernel + mask 处理，避免两套 kernel
```

**MoE Expert 打包**：

```
Expert e 对应权重矩阵（M_e × K）：
  qweight_int2[e]: shape = (K//16, M_e), dtype = int32
  scales[e]:       shape = (K//128, M_e), dtype = float16

各 expert 独立 buffer，按 128-byte 对齐
Expert 指针：
  uint64_t qweight_ptrs[n_experts];  // 各 expert qweight 起始地址
  uint64_t scale_ptrs[n_experts];    // 各 expert scale 起始地址
  uint32_t expert_rows[n_experts];   // 各 expert 行数（M_e）

使用 grouped GEMM（CUTLASS GroupedGemm）处理所有活跃 expert：
  每个 expert 作为一个独立 GEMM problem，共用同一 CUDA kernel launch
```

---

### 5. QTIP（TCQ Trellis）— 计算型解码，无显式 codebook

**核心存储理念**：QTIP 不存储量化值，而存储 trellis 符号；解码时通过 LCG 或小 codebook 计算 Gaussian 量化网格点。

**Trellis Segment 存储单元**（per 16 weights，3-bit 1MAD 模式）：

```
Trellis segment（16 weights，1MAD 模式）：
  trellis_syms: 3 个 uint16（48 bits = 16 × 3-bit symbols）
  scale:        1 × FP16（sub-segment scale）
总计：8 bytes / 16 weights = 4 bytes/weight overhead
实际 bpw = 3 + 16/16 = 4 bpw（含 scale，比 3-bit 略高）

HYB 模式（hybrid 计算+查表）：
  trellis_syms: 3 × uint16
  lcg_seed:     1 × uint32（per block）
  2KB codebook 在 GPU shared memory 中，不随权重存储
```

**矩阵权重布局（行 tile 主序）**：

```
每行按 tile_K 个 weights 为一组（tile_K = 256 = 16 segments）：
  Row tile 结构：16 个连续 trellis segment（16×8 bytes = 128 bytes）

矩阵 (M × K) 存储：
[row0_tile0][row0_tile1]...[row0_tile_{K/256-1}]
[row1_tile0][row1_tile1]...

每行首地址对齐：128 bytes（一个 cache line = 1 tile）
```

**GEMV 最优分配（Decode，1MAD 模式）**：

```
每 warp 处理 1 个 trellis segment（16 weights）：
  Thread 0–2：加载 trellis_syms（3 × uint16 = 6 bytes）
  Thread 0：  加载 scale（2 bytes），warp 广播
  Thread 0：  获取 LCG seed（block 级，从 segment 头部读取）
              计算 Gaussian grid：x = LCG(seed + thread_id) * scale
  Thread i（i<16）：decode trellis symbol i → grid 偏移 → weight[i] = grid_val
  warp_reduce → 累加到寄存器

Bitshift trellis 并行关键：
  每个 symbol[i] 的解码依赖 symbol[i-1] 的 trellis state
  Bitshift 变换使 state 可通过 bit shift 独立计算，无需 serial Viterbi
  → 16 symbols 可由 16 threads 并行解码
```

**GEMM 最优布局（Prefill）**：

QTIP 的 trellis 格式不能直接用于 WMMA；推荐 **dequant-first** 方案：

```
Dequant-first 策略：
  1. 在 GEMM kernel 内部，先 dequant 一个 K-tile（256 weights × tile_M 行）
     → 存放在 shared memory（FP16 格式，256 × tile_M × 2 bytes）
  2. 使用标准 FP16 WMMA 完成矩阵乘
  3. 共享 codebook（HYB 模式）也放在 shared memory

Shared Memory 分配（单个 thread block）：
  dequant_buffer: tile_M × tile_K × 2 bytes（FP16）   // 例如 16×256×2 = 8KB
  codebook_qtip:  2 KB（HYB 模式，固定）
  合计：~10 KB per thread block（在 H100 48KB smem 范围内）
```

**MoE Expert 打包**：

```
每个 expert 的权重按行 tile 连续存放：
  Expert e：[M_e × (K/256)] 个 trellis segment groups
  段边界对齐：128 bytes（1 trellis tile = 1 cache line）
  Expert 边界：256 bytes

LCG seed 管理：
  每个 K-block（256 weights）对应 1 个 uint32 seed
  seed 表：shape = (M, K//256)，与权重同 expert 分组存放
  或：seed 内嵌在每个 row tile 的头部（无需单独 seed 表）
```

---

### 6. Layout 决策对比矩阵

| 量化方法 | 基本 decode 单元 | GEMV 布局 | GEMM 布局 | 是否需要预转换 | 关键 smem 需求 |
|---|---|---|---|---|---|
| **Q3_K** | 256W super-block | Row-major blocks，行首 128B 对齐 | 展开为 INT4，CUTLASS interleaved | 是（INT4 预处理） | 最小 |
| **IQ3_XXS** | 32W sub-block | Row-major blocks，sub-block 连续 | 字段拆分，K-major 重排 | 推荐 | Codebook 1KB（constant） |
| **IQ3_S** | 32W sub-block | Row-major blocks，qh/signs 独立 | 字段拆分，K-major 重排 | 推荐 | Codebook 2KB（constant） |
| **SignRoundV2 INT2** | 128W group | Transposed (N, K//16)，行主序 | CUTLASS interleaved (K//128,N,128) | 否（直接用） | Scale 缓存 |
| **SignRoundV2 混合** | 128W group（分精度） | 同上，双 kernel 或 mask 标记 | 同上，mask fusion | 否 | 精度 mask |
| **QTIP 1MAD/3INST** | 16W segment | Row-major trellis tiles，128B/tile | Dequant-first → FP16 smem | 是（实时 dequant） | 8–10 KB/block（dequant buffer） |
| **QTIP HYB** | 16W segment | 同上 | 同上 + 2KB codebook | 是 | 10 KB/block |

---

### 7. MoE 通用 Expert 权重打包建议

```c
// OpenVINO MoE GPU 量化权重统一描述结构
struct QuantExpert {
    void*    qweight;      // 量化权重 buffer 起始指针
    void*    scales;       // scale buffer
    void*    zeros;        // zero point（可选）
    uint32_t n_rows;       // 输出维度 M_e
    uint32_t n_cols;       // 输入维度 K
    uint32_t quant_type;   // OV_QUANT_Q3K / OV_QUANT_IQ3S / OV_QUANT_INT2 等
    uint32_t group_size;   // dequant group size（对 Q3_K = 256，INT2 = 128）
};

// 连续 buffer 内 expert 间距对齐：256 bytes（避免 cache set aliasing）
// 推荐：使用统一 allocator，按 n_experts × max_expert_size 预分配
```

**Decode（GEMV）Expert 路由模式**：

```
// top-k 路由：每 token 激活 k 个 expert
// → 使用 expert 指针数组，每次 GEMV 指定 expert_ptr
for each active expert e in top_k_indices:
    gemv_dequant(expert[e].qweight, x_routed, y_partial, expert[e])
y_final = gating_weights[e] * y_partial[e]  // weighted sum
```

**Prefill（GEMM）Expert 路由模式**：

```
// 将 token 按激活 expert 分组（token routing/sorting）
// 使用 grouped GEMM：
grouped_gemm_config[n_active_experts]:
  - problem_size[e] = (batch_e, M_e, K)  // batch_e = tokens routed to expert e
  - A[e] = X_routed[e]                   // activation slice
  - B[e] = expert[e].qweight             // weight（dequant in kernel）
  - C[e] = Y_partial[e]                  // output

// 推荐实现：CUTLASS GroupedGemm 或 OpenVINO custom grouped_matmul kernel
```

---

## 八、SignRoundV2 解压算法详细解析

> 基于 arXiv:2512.04746 和 [intel/auto-round](https://github.com/intel/auto-round) 源码的深度分析

---

### 8.1 算法全流程总览

SignRoundV2 是**纯标量 PTQ**，其量化与解压管线可分为五个阶段：

```
FP16 权重 W
    │
    ▼ 【Phase 1】量化参数搜索（pre-tuning scale search）
    │  候选 scale s_i = max(|W|) / (2^(bit-1)) + ε_i，ε_i ∈ [-0.9, 0.9, step=0.01]
    │  搜索最小化 importance-matrix 加权误差的最优 s_init
    │
    ▼ 【Phase 2】Sign-gradient rounding 优化（主训练）
    │  引入可学习参数：rounding bias v（per-weight），scale α（per-group），zero β
    │  qdq(W) = s · clip(⌊W/s + v⌋, n, m)   —— 前向
    │  梯度通过 STE（Straight-Through Estimator）反传到 v, α, β
    │  优化 200 步（标准）/ 500 步（Ours*），使用 sign-SGD
    │
    ▼ 【Phase 3】DeltaLoss 灵敏度评估 + 动态规划 bit 分配
    │  ΔL_i(b) ≈ |g_wq ∘ (W_f - W_q)|  per layer per bit-width
    │  动态规划求解：在 bit 预算约束下最小化 ΣΔL_i(b_i)
    │  → 每层得到 bit 分配：W2G128 或 W4G128
    │
    ▼ 【Phase 4】权重打包（packing）存储
    │  将优化后的整数权重 q 按 GPTQ 兼容格式打包为 int32 tensor
    │
    ▼ 【Phase 5】推理时解压（dequantization）
       从 int32 qweight 提取各 weight 的整数值 → 乘以 scale → 执行 matmul
```

---

### 8.2 量化数学公式

#### 基础量化（SignRoundV1 继承，V2 沿用）

$$
q = \text{clip}\!\left(\left\lfloor \frac{W}{s} + v \right\rceil,\; n,\; m\right)
$$

$$
\hat{W} = s \cdot (q - z)
$$

其中：
- $s$：group scale（FP16），每 `group_size` 个权重共享一个
- $v$：per-weight rounding bias（训练后丢弃，不存储）；推理时 $v$ 效果已固化到 $q$ 中
- $z$：zero point（整数），对称量化时 $z = 0$
- $n, m$：量化范围，INT2 时 $n = -2, m = 1$；INT4 时 $n = -8, m = 7$（有符号）

#### 对称量化的简化（默认配置）

$$
q = \text{clip}\!\left(\text{round}\!\left(\frac{W}{s}\right),\; -2^{b-1},\; 2^{b-1}-1\right)
$$

$$
\hat{W} = s \cdot q
$$

Scale 初始化：

$$
s_{\text{init}} = \frac{\max(|W|)}{2^{b-1} - 1} \cdot \alpha, \quad \alpha \in [0.5, 1.5]
$$

#### 混合精度（3-bit 平均）实现方式

SignRoundV2 **不存在真正的 3-bit 数据类型**，"3-bit" 是通过层级混合实现的：

```
模型总权重参数量 = P_total

DeltaLoss 动态规划分配结果：
  敏感层（少数）→ W4G128：qweight 用 INT4 打包（4 bit/weight）
  非敏感层（多数）→ W2G128：qweight 用 INT2 打包（2 bit/weight）

平均比特数（忽略 scale/zp overhead）：
  avg_bits = (P_int4 × 4 + P_int2 × 2) / P_total ≈ 3 bit（目标）

Scale overhead（实际 bpw 增量）：
  INT2G128：+ 16 bit / 128 weights = +0.125 bpw → 实际 2.125 bpw
  INT4G128：+ 16 bit / 128 weights = +0.125 bpw → 实际 4.125 bpw
  平均：约 3.125 bpw（混合后）
```

---

### 8.3 权重打包格式（GPTQ 兼容）

auto-round 使用与 GPTQ 完全兼容的 int32 打包格式（源自 `qlinear_tritonv2.py`）：

#### INT2 打包（bits=2）

```
qweight: shape = (K // 16, N), dtype = int32
  ← 权重矩阵 W 形状为 (N, K)，先按 group 量化，再沿 K 方向每 16 个打包成 1 个 int32

打包方式（LSB 到 MSB，低 K 地址在低 bit）：
  qweight[k//16][n] = q[n][k]      <<  0  (bits 1:0)
                    | q[n][k+1]    <<  2  (bits 3:2)
                    | q[n][k+2]    <<  4  (bits 5:4)
                    | ...
                    | q[n][k+15]   << 30  (bits 31:30)

注意：已经 .t().contiguous()，即转置后连续存储
  → qweight[k_packed][n] 表示：输出维度 n，输入 K 位置 k_packed*16 开始的 16 个权重
  → 内存中 n 维度连续（column-major from K perspective）
```

#### INT4 打包（bits=4）

```
qweight: shape = (K // 8, N), dtype = int32
  每 int32 存 8 个 INT4 权重（bits=4，32/4=8）

打包方式：
  qweight[k//8][n] = q[n][k]    <<  0  (bits 3:0)
                   | q[n][k+1]  <<  4  (bits 7:4)
                   | ...
                   | q[n][k+7]  << 28  (bits 31:28)
```

#### Scale 和 Zero Point 格式

```
scales: shape = (K // group_size, N), dtype = float16
  scales[g][n] = scale for weights W[n][g*group_size : (g+1)*group_size]

qzeros: shape = (K // group_size, N // (32 // bits)), dtype = int32
  Zero points 也按 bits 打包（同 qweight 格式）：
  INT2: qzeros shape = (K//group_size, N//16)
  INT4: qzeros shape = (K//group_size, N//8)

  qzeros[g][n//16] 存放 N 方向 16 个输出通道的 zero point（INT2 对称量化时全为 0）
```

#### 整体内存布局示意

```
对于 W2G128（K=4096, N=4096）：

  qweight:  (4096//16, 4096) = (256, 4096) int32  → 256 * 4096 * 4 = 4 MB
  scales:   (4096//128, 4096) = (32, 4096) fp16  → 32 * 4096 * 2 = 256 KB
  qzeros:   (32, 256) int32                      → 32 * 256 * 4 = 32 KB（对称时可省略）
  总计：约 4.3 MB

对比 FP16：4096 * 4096 * 2 = 32 MB
压缩比：32 / 4.3 ≈ 7.4×（接近 8×，overhead 来自 scale）
```

---

### 8.4 解压（Dequantization）算法

#### 标准解压公式

对于 int32 打包的 qweight，提取单个权重 $\hat{W}[n][k]$ 的步骤：

```python
# INT2 解压伪代码
def dequant_int2(qweight, scales, qzeros, k, n, group_size=128):
    k_packed = k // 16                    # qweight 行索引
    bit_offset = (k % 16) * 2            # 在 int32 内的 bit 位置

    # 1. 从 int32 中提取 2-bit 整数（无符号）
    q_uint = (qweight[k_packed][n] >> bit_offset) & 0x3   # → 0, 1, 2, 3

    # 2. 提取 zero point（对称时 zp=2，即区间中心）
    g = k // group_size
    zp_packed = qzeros[g][n // 16]
    zp_offset = (n % 16) * 2
    zp = (zp_packed >> zp_offset) & 0x3   # zp ∈ {0,1,2,3}

    # 3. 计算有符号偏移值（居中到 [-2, 1]）
    q_signed = q_uint - zp               # 对称时：zp=2 → q_signed ∈ {-2,-1,0,1}

    # 4. 乘以 scale
    scale = scales[g][n]                 # FP16
    W_approx = q_signed * scale          # FP16 multiply

    return W_approx

# INT4 同理，bit_offset = (k % 8) * 4，mask = 0xF，zp range = {0..15}
```

#### Triton 核心解压 Kernel 模式

auto-round 的 Triton 解压 kernel（`triton_utils/dequant`）按 tile 批量解压：

```python
# 概念性 Triton kernel（基于 Triton v2 的 dequant 模式）
@triton.jit
def dequant_kernel(
    qweight_ptr, scales_ptr, qzeros_ptr,
    output_ptr,
    K, N, group_size, bits,
    BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr
):
    # 每个 thread block 处理 (BLOCK_K, BLOCK_N) tile
    pid_k = tl.program_id(0)
    pid_n = tl.program_id(1)

    k_start = pid_k * BLOCK_K
    n_start = pid_n * BLOCK_N

    # 加载 tile 对应的 int32 packed weights
    k_packed_start = k_start // (32 // bits)
    k_packed_range = tl.arange(0, BLOCK_K // (32 // bits))
    n_range = tl.arange(0, BLOCK_N)

    qw = tl.load(qweight_ptr + (k_packed_start + k_packed_range[:, None]) * N
                              + (n_start + n_range[None, :]))  # shape: (BLOCK_K//(32//bits), BLOCK_N)

    # 提取各 bit 位
    shift = tl.arange(0, 32 // bits) * bits              # [0, 2, 4, ..., 30] for INT2
    mask = (1 << bits) - 1                               # 0x3 for INT2

    # broadcast: [K_packed, N] × [32//bits] → [K_packed * 32//bits, N] = [BLOCK_K, N]
    q_uint = (qw[:, :, None] >> shift[None, None, :]) & mask   # shape: (K_packed, N, 16)
    q_uint = q_uint.reshape(BLOCK_K, BLOCK_N)                   # shape: (BLOCK_K, BLOCK_N)

    # 加载 scale 和 zero point（按 group 对齐）
    g_range = (k_start + tl.arange(0, BLOCK_K)) // group_size
    scale = tl.load(scales_ptr + g_range[:, None] * N + (n_start + n_range[None, :]))
    # ... zp 同理 ...

    # 解压
    W_approx = (q_uint.to(tl.float16) - zp) * scale

    tl.store(output_ptr + ..., W_approx)
```

#### 推理时的两种模式

**模式 A：Dequant-then-matmul（TritonV2 默认）**

```
1. 整层权重一次性或按 tile 解压为 FP16
2. 使用 torch.matmul(x, W_fp16)
3. 适合 prefill（batch > 1），吞吐优化
4. 内存开销：临时 FP16 weight buffer（N×K × 2 bytes）
```

**模式 B：Fused dequant + matmul（Marlin/ExLLaMA kernel）**

```
1. 在 GEMM kernel 内部，每个 thread block 按 tile 实时解压
2. 解压结果存 shared memory，立即用于 WMMA 计算
3. 适合 decode（batch=1），减少显存带宽压力
4. 内存开销：仅 shared memory 内的 tile buffer（约 16KB 以内）
```

---

### 8.5 混合精度（INT2 + INT4）的推理实现

#### 层级 bit 分配存储

DeltaLoss 分配的结果固化在模型 checkpoint 中：

```python
# 每层独立存储自己的精度格式（通过 config 标记）
layer_config = {
    "model.layers.0.mlp.gate_proj":   {"bits": 2, "group_size": 128},
    "model.layers.0.mlp.down_proj":   {"bits": 4, "group_size": 128},  # 敏感层
    "model.layers.0.mlp.up_proj":     {"bits": 2, "group_size": 128},
    "model.layers.1.mlp.gate_proj":   {"bits": 2, "group_size": 128},
    ...
}

# 对应的 qweight 形状不同：
# bits=2: qweight.shape = (K//16, N)  → 更小
# bits=4: qweight.shape = (K//8,  N)  → 较大
```

#### 推理 dispatch 逻辑

```python
class QuantLinear(nn.Module):
    def forward(self, x):
        if self.bits == 2:
            # INT2 dequant path
            W = dequant_int2(self.qweight, self.scales, self.qzeros, ...)
        elif self.bits == 4:
            # INT4 dequant path（相同接口，不同 bit 宽）
            W = dequant_int4(self.qweight, self.scales, self.qzeros, ...)

        return x @ W.T + self.bias
```

同一 kernel（如 Triton dequant kernel）通过 `bits` 参数分支处理，无需两套 kernel：

```python
# 关键参数复用：
n_per_int32 = 32 // bits  # INT2: 16, INT4: 8
mask = (1 << bits) - 1    # INT2: 0x3, INT4: 0xF
bit_shifts = torch.arange(0, n_per_int32) * bits  # INT2: [0,2,4,...,30]
```

---

### 8.6 DeltaLoss 灵敏度评估详解

DeltaLoss 是 SignRoundV2 选择 bit 分配的核心，与解压直接相关（决定哪些层用 INT2，哪些用 INT4）：

#### 公式推导

**原始一阶 Taylor 展开**（weight + activation 联合）：

$$
\Delta\mathcal{L} \approx \left|g_{A_q} \circ (A_f - A_q)\right| + \left|g_{W_q} \circ (W_f - W_q)\right|
$$

**Weight-only（W2A16）简化版**：

$$
\Delta\mathcal{L} \approx \left|g_{W_q} \circ (W_f - W_q)\right|
$$

其中：
- $g_{W_q} = \partial\mathcal{L}/\partial W_q$：损失对量化权重的梯度（通过前向一次反向一次得到）
- $W_f - W_q$：量化误差 = $(W_f - \hat{W}) = W_f - s \cdot q$（数值可直接计算）
- $\circ$：Hadamard 积（element-wise）

#### 计算流程

```python
def compute_delta_loss(model, layer, bits_list, calibration_data):
    delta_loss_per_bits = {}

    for bits in bits_list:  # e.g., [2, 4]
        # 1. 量化该层并计算量化误差
        W_q = quantize(layer.weight, bits=bits, group_size=128)
        quant_error = layer.weight - W_q  # W_f - W_q

        # 2. 前向传播，得到量化权重的梯度
        model.eval()
        output = model(calibration_data)  # 使用量化后的 W_q
        loss = cross_entropy(output, targets)
        loss.backward()
        g_Wq = layer.weight.grad.clone()  # ∂L/∂W_q

        # 3. DeltaLoss = |gradient × quantization_error|
        delta_loss = (g_Wq * quant_error).abs().sum()
        delta_loss_per_bits[bits] = delta_loss.item()

    return delta_loss_per_bits  # e.g., {2: 0.045, 4: 0.012}
```

#### 动态规划 bit 分配

```python
def dp_bit_allocation(delta_loss_table, target_avg_bits, layer_params):
    """
    delta_loss_table[i][b] = DeltaLoss of layer i at b bits
    target_avg_bits: e.g., 3.0
    layer_params[i]: parameter count of layer i
    """
    # 等价于 0-1 背包问题（每层选一个 bit 宽）
    # 约束：Σ(b_i × P_i) ≤ target_avg_bits × P_total
    # 目标：min Σ DeltaLoss_i(b_i)

    # 用 DP 求解（O(n × bit_budget)，实际 << 1 秒）
    ...
    return {layer_i: assigned_bits_i}
```

实验结果（Table 4，W2G128/W4G128 平均 3 bit）：

| 方案 | Llama3.1-8B-I | Qwen2.5-7B-I | Qwen3-8B |
|---|---|---|---|
| Head 4-bit（固定头部高精度） | 31.98 | 31.96 | 32.70 |
| Tail 4-bit（固定尾部高精度） | 60.58 | 45.36 | 37.98 |
| **SignRoundV2 DeltaLoss DP** | **61.48** | **48.62** | **40.58** |

DeltaLoss 分配明显优于简单的头/尾固定策略。

---

### 8.7 解压性能分析

#### 解压操作数（per weight）

| 操作 | INT2 | INT4 | FP16（参考） |
|---|---|---|---|
| 位移/掩码（unpack） | 1 shift + 1 AND | 1 shift + 1 AND | 0 |
| 减 zero point | 1 INT sub | 1 INT sub | 0 |
| 乘 scale | 1 FP16 mul | 1 FP16 mul | 0 |
| 类型转换 | 1 INT→FP16 | 1 INT→FP16 | 0 |
| **合计（ops/weight）** | **~4** | **~4** | **0** |

解压操作数相同（INT2 vs INT4），开销主要差异在**内存带宽**：
- INT2 权重加载带宽 = FP16 的 1/8（内存访问节省 8×）
- INT4 权重加载带宽 = FP16 的 1/4（内存访问节省 4×）

#### Decode（GEMV）速度估算

以 RTX 6000 Ada 为例（内存带宽 960 GB/s，FP16 matmul ~220 TFLOPS）：

```
INT2 GEMV 理论加速（内存瓶颈）：
  FP16 bandwidth: 960 GB/s
  INT2 weight load: 1/8 of FP16 → 有效带宽节省 ~7.875x（含 scale 开销）
  scale 带宽: (K/128) × 2 bytes / 128 = 1/64 of FP16 → negligible
  净加速: ~6–7× vs FP16 GEMV（与 INT4 约 3–3.5× 相比快 2×）

INT4 GEMV 理论加速：~3.5× vs FP16

实测（Marlin/ExLLaMA2 kernel）：
  INT4G128: 2–3× speedup at batch=1
  INT2G128: 3–5× speedup at batch=1（解压 overhead 抵消部分带宽收益）
```

#### Prefill（GEMM）效率

GEMM 为计算密集，INT2/INT4 收益主要来自激活 cache 效率，加速相对有限：

```
典型 batch=64 prefill：
  FP16 GEMM: compute-bound，接近 Tensor Core 峰值
  INT4 GEMM（CUTLASS INT4）: ~1.3–1.6× speedup（Tensor Core INT4 峰值更高）
  INT2 GEMM: 需要先 dequant 到 INT4，额外 overhead → ~1.2× speedup
```

---

### 8.8 量化参数初始化（Pre-tuning Scale Search）详解

SignRoundV2 的关键创新之一：**先搜索好的 scale 初值，再做梯度优化**。

#### 搜索目标（类 importance-matrix）

$$
\min_{s \in S}\; \frac{1}{N}\sum_{i=1}^{N}\left[(W_f - \hat{W}_q(s)) \circ \max(\bar{A})\right]_i^2
$$

其中 $\max(\bar{A})$ 是从 calibration 数据统计的 per-input-channel 最大激活值（体现该通道的重要性），$S$ 是候选 scale 集合。

#### 搜索过程

```python
# 候选 scale 生成
base_scale = max(abs(W)) / (2 ** (bits - 1) - 1)  # 默认 scale
epsilon_range = torch.arange(-0.9, 0.9, 0.01)      # 181 个候选
candidates = [base_scale * (1 + eps) for eps in epsilon_range]

# 对每个候选 scale 计算 importance-weighted 量化误差
best_scale = base_scale
best_loss = float('inf')

for s in candidates:
    W_q = quantize(W, scale=s, bits=bits)
    # importance-weighted MSE
    loss = ((W - W_q) * max_activation_per_channel) ** 2
    loss = loss.mean()

    if loss < best_loss:
        best_loss = loss
        best_scale = s

# 用 best_scale 初始化 s_init，后续由 α ∈ [0.5, 1.5] 进一步微调
s_init = best_scale
```

#### 初始化的实际影响（Table 5，W2A16G64）

| 模型 | 无初始化 (Avg) | 有初始化 (Avg) | 提升 |
|---|---|---|---|
| Qwen3-8B（ARC-c） | 34.90 | 43.69 | **+8.79%** |
| Qwen3-8B（Hella.） | 47.18 | 54.61 | **+7.43%** |
| Llama3.1-8B-I（Hella.） | 52.57 | 60.41 | **+7.84%** |

说明：对于 2-bit 量化，初始化质量对最终精度影响极大（+5–9% 准确率提升）。

---

### 8.9 Sign-gradient Descent 优化原理

#### 为什么用 sign gradient 而非 Adam？

普通量化优化中，rounding bias $v \in [0,1]$ 的梯度非常小且不稳定（因为 $\lfloor \cdot \rceil$ 的 STE 梯度是矩形函数）。Sign-SGD 解决了这个问题：

$$
v_{t+1} = v_t - \eta \cdot \text{sign}(\nabla_v \mathcal{L})
$$

每步更新量固定为 $\eta$（learning rate），避免了 Adam 的自适应 lr 对量化参数的过拟合。

```python
# SignRound 优化循环（简化）
for step in range(max_steps):  # 200 steps
    output_quant = forward_with_qdq(x, W, v, alpha, beta)
    output_fp = forward_fp16(x, W_original)

    # 重建误差（block-wise，对同一 transformer block 的输出）
    loss = mse_loss(output_quant, output_fp)
    loss.backward()

    # sign-gradient update（不是 Adam）
    with torch.no_grad():
        v.data -= lr * v.grad.sign()     # rounding bias
        alpha.data -= lr * alpha.grad.sign()  # scale multiplier
        beta.data -= lr * beta.grad.sign()    # zero point multiplier

        # 约束 alpha 在合理范围
        alpha.data.clamp_(0.5, 1.5)

    optimizer.zero_grad()

# 量化完成后：v, alpha, beta 丢弃
# 只保存量化后的整数权重 q = round(W/s + v) 和 scale s = s_init * alpha
```

#### 为什么能提升 2-bit 精度？

GPTQ 和 RTN 在 2-bit 时精度极差（Llama2-7B: 41.56%）的核心原因是**舍入误差的累积效应**：

```
RTN（四舍五入）：
  每个权重的 rounding error ∈ [-0.5s, 0.5s]
  layer 输出误差 ∝ Σ_k error_k * x_k（可能同向累积）

SignRound（优化 rounding）：
  选择 ⌊W/s⌋ 或 ⌈W/s⌉（由 v 决定）
  使得 layer 输出的 block 重建误差最小
  → 误差不同向，相互抵消，显著降低最终 loss
```

这就是为什么在 Llama2-7B W2G128：
- RTN: 41.56% → SignRoundV2: 57.88%（提升 **16.3%**）
- 代价仅为额外的 2.5h 量化时间

---

### 8.10 OpenVINO 实现要点（3-bit MoE 场景）

基于以上分析，OpenVINO GPU 插件实现 SignRoundV2 解压的关键点：

#### 量化格式适配

```cpp
// 支持 OpenVINO 已有的 INT4 weight-only 路径
// SignRoundV2 W4G128 可直接复用现有 INT4 GEMM kernel

// 新增 W2G128 支持需要：
// 1. 新的打包格式：qweight(K//16, N) int32（vs INT4 的 K//8）
// 2. 相同的 scale 格式：scales(K//128, N) fp16
// 3. 新的解压 kernel 或统一参数化的解压函数
```

#### 解压 kernel 设计要点

```cpp
// 统一解压 kernel（INT2 和 INT4 共享，通过 bits 参数区分）
__device__ half dequant_weight(
    const int* qweight,
    const half* scales,
    const int* qzeros,       // NULL for symmetric
    int k, int n,
    int K, int N,
    int group_size, int bits
) {
    int n_per_int = 32 / bits;
    int k_packed = k / n_per_int;
    int bit_offset = (k % n_per_int) * bits;
    int mask = (1 << bits) - 1;

    // 1. Unpack quantized value
    int q_uint = (qweight[k_packed * N + n] >> bit_offset) & mask;

    // 2. Zero point（对称量化时 zp = 2^(bits-1)，即无符号→有符号）
    int zp = 1 << (bits - 1);   // symmetric: zp=2 for INT2, zp=8 for INT4

    // 3. Dequantize
    int g = k / group_size;
    half scale = scales[g * N + n];
    return __hmul(scale, __int2half_rn(q_uint - zp));
}
```

#### MoE Expert 路由适配

```cpp
// SignRoundV2 的 expert weight buffer 结构
struct SignRoundExpert {
    int32_t* qweight_int2;   // shape: (K//16, M_e)，INT2 层
    int32_t* qweight_int4;   // shape: (K//8,  M_e)，INT4 层（非空时用 INT4）
    half*    scales;         // shape: (K//128, M_e)
    int32_t* qzeros;         // 对称时 NULL
    uint8_t  bits;           // 2 or 4
    uint32_t group_size;     // 128
    uint32_t n_rows;         // M_e
    uint32_t n_cols;         // K
};

// Decode 路由：根据 expert bits 选择 GEMV kernel
if (expert.bits == 2) {
    gemv_dequant_int2(expert.qweight_int2, x, y, expert.scales, expert.n_rows, K);
} else {
    gemv_dequant_int4(expert.qweight_int4, x, y, expert.scales, expert.n_rows, K);
}
```

#### 与现有 OpenVINO INT4 路径的差异对比

| 维度 | OpenVINO INT4（已有） | SignRoundV2 W2G128（新增） |
|---|---|---|
| qweight 形状 | (K//8, N) int32 | (K//16, N) int32 |
| scale 形状 | (K//128, N) fp16 | 相同 |
| 解压公式 | (q - zp) × s | 相同（bits=2） |
| GEMV kernel | 已有 | 需新增（bits=2 路径） |
| GEMM kernel | CUTLASS INT4 | 先 dequant→INT4，或直接 INT2 CUTLASS |
| 精度（70B） | ~70–72% | 68.82%（Ours*） |
| bpw | 4.125 | 2.125 |

---

## 第九节：LoRA 感知量化算法族与 3-bit PTQ 增强（NNCF LoRA QAT、3bit SE、3bit SE+AWQ）

> 本节覆盖 Jira CVS-182951 中列出的新增算法：NNCF LoRA QAT (3bit)、3bit SE、3bit SE+AWQ，以及 SKILL.md 中新增论文：IR-QLoRA、ParetoQ、LR-QAT、ApiQ。

---

### 9.0 背景：LoRA + 量化的两大范式

LoRA 与量化结合有两条主流路线，区别在于 LoRA 适配器与量化函数的位置关系：

**路线 A：PTQ-first + LoRA-finetune（后量化 + 适配器微调）**

先做 PTQ 得到低比特权重，再冻结量化权重，在其上训练 LoRA 适配器。推理时需要额外的 LoRA 计算开销：

```
推理：y = dequant(W_q) · x + α/r · A·B · x
代表：QLoRA, LoftQ, PiSSA, IR-QLoRA, ApiQ
```

**路线 B：QAT + LoRA 联合训练（量化感知训练 + 低秩参数化）**

将 LoRA 适配器置于量化函数**内部**，训练后直接融合到整数权重格点，推理等同于纯 PTQ，无额外 LoRA 开销：

```
训练：W_q = FQ(W₀ + α/r·A·B)
训练后：W_fused = W₀ + α/r·A·B → 量化 → W_Z（纯整数矩阵）
推理：y = dequant(W_Z) · x
代表：LR-QAT, NNCF LoRA QAT
```

NNCF LoRA QAT 属于路线 B。两条路线的关键公式区别：

| 方法 | 公式 | 推理 LoRA 开销 |
|---|---|---|
| LoftQ / QLoRA | `W_q = FQ(W₀) + A·B` | 有（额外矩阵乘法） |
| LR-QAT / NNCF LoRA QAT | `W_q = FQ(W₀ + α/r·A·B)` | **无**（融合后消失） |

---

### 9.1 LoftQ（ICLR 2024）

**来源**：arXiv:2310.08659，Microsoft Research

**核心问题**：QLoRA 默认初始化（B=0，A 高斯随机）导致训练起点差——初始时 `y = FQ(W)·x`，与原始 `y = W·x` 的量化误差在 ≤3-bit 时无法通过梯度快速弥合。

**解法：迭代 SVD 交替优化**：
```
初始化：Q⁽⁰⁾ = FQ(W)
for t = 1..T:
    A⁽ᵗ⁾, B⁽ᵗ⁾ ← top-r SVD( W - Q⁽ᵗ⁻¹⁾ )   # 从量化残差初始化适配器
    Q⁽ᵗ⁾ ← FQ( W - A⁽ᵗ⁾·B⁽ᵗ⁾ )               # 量化剩余残差
```

最终：`W ≈ Q⁽ᵀ⁾ + A⁽ᵀ⁾·B⁽ᵀ⁾`，保证初始时模型输出尽可能接近原始 FP16 模型。

**SKILL.md 注记**：与标准 LoRA（B=0 导致 AB=0，梯度信号弱）相比，LoftQ 的 A 和 B 都从有信号的残差 SVD 初始化，显著加快收敛。

**3-bit 效果**（vs QLoRA，LLaMA-7B MMLU）：+1.1% ~ +2.8%

**代码**：`github.com/yxli2123/LoftQ`

---

### 9.2 PiSSA（NeurIPS 2024 spotlight）

**来源**：arXiv:2404.02948，PKU & NeurIPS 2024

**核心思想**：不同于 LoftQ 从"残差"初始化适配器，PiSSA 将 W 的**主奇异值分量**（最重要部分）给适配器，只量化**非主分量**（残差）。

**公式**：
```python
U, S, Vt = torch.linalg.svd(W)          # 全 SVD

# 主成分（前 r 个）→ 适配器初始化（保留在 FP16，可微调）
A_init = U[:, :r] * (S[:r] ** 0.5)
B_init = (S[:r] ** 0.5).unsqueeze(-1) * Vt[:r, :]

# 残差（剩余成分）→ 量化（存储低比特）
W_res = W - A_init @ B_init
W_q = FQ(W_res)                          # 量化非主成分
```

**推理**：`y = dequant(W_q)·x + A·B·x`（与 QLoRA 结构相同，但 A/B 有意义初始化）

**QPiSSA 3-bit 效果**（LLaMA-3-70B GSM8K）：
- QLoRA: 81.73%
- QPiSSA: **86.05%**（+4.3%）

**限制**：A/B 在推理时不能融合进量化格点（有 LoRA 推理开销）

**代码**：`github.com/GraphPKU/PiSSA`

---

### 9.3 IR-QLoRA（ICML 2024）

**来源**：arXiv:2402.05445，ICML 2024

**核心思想**：从信息论角度改进量化 + LoRA 训练，通过最大化量化权重的信息熵来恢复量化损失的信息。

#### 技术 1：信息校准量化 ICQ（Information Calibration Quantization）

在 NormalFloat 量化器中引入可学习偏移常数 τ，最大化量化权重的信息熵：

$$
\hat{W}^{\text{ICQ}}_{NF_k} = NF_k\left(\frac{W - \tau^*}{\text{absmax}(W - \tau^*)}\right)
$$

- τ 初始化为 `median(W)`（利用 Gaussian 假设最大化格点利用）
- 二步搜索：先初始化，再在 `[τ₀ - λσ, τ₀ + λσ]` 线性搜索最优 τ*
- 双量化存储（τ₁_FP8, τ₂_FP16），额外内存仅 +2% 

**完整解压公式**：
$$
\hat{W}^{\text{ICQ}}_{FP16} = \hat{W}^{\text{ICQ}}_{NF_k} \cdot \text{dequant}(s_1^{FP8}, s_2^{FP16}) + \text{dequant}(\tau_1^{FP8}, \tau_2^{FP16})
$$

#### 技术 2：信息弹性连接 IEC（Information Elastic Connection）

通过零参数跳跃连接增强 LoRA 信息流动（每层仅增加 2 个可学习标量 β₁, β₂）：

```
U1(x) = x·l1 + β1 · (r/h · Σᵢ x[i·h/r : (i+1)·h/r])  # 分组平均 + 重复
U2(x') = x'·l2 + β2 · repeat(x', o/r)                  # 重复扩展到输出维度
y = y_ICQ + α · U2(U1(x))
```

训练后 β₁, β₂ 可融合进 l₁, l₂，推理无额外开销。

**3-bit 效果**（LLaMA-7B，Alpaca 微调，MMLU）：

| 方法 | 3-bit MMLU |
|---|---|
| QLoRA | 37.8% |
| QA-LoRA | 37.4% |
| IR-QLoRA | **38.4%** |

**代码**：`github.com/htqin/ir-qlora`

---

### 9.4 ApiQ（ICML 2024）

**来源**：arXiv:2402.05147，University of Amsterdam

**核心创新**：不最小化权重误差 `‖W - (Q+AB)‖`，而是最小化**激活误差** `‖X·W - Xq·(Q+AB)‖`，从而控制量化误差跨层累积传播。

**层序优化（ApiQ-lw）**：
```python
for each linear layer in forward order:
    Y = X @ W                          # 原始激活（target）
    for epoch in range(epochs):
        # 联合优化 Q（量化权重）+ A, B（LoRA 适配器）+ γ, β（量化 clip 范围）
        min_{Q, A, B, γ, β}  ‖Y - Xq @ (Q + A@Bᵀ)‖
    Xq = Xq @ (Q + A@Bᵀ)             # 量化层输出，传递给下一层
```

**块级优化（ApiQ-bw）**：以 transformer block 为单位，兼容 DoRA/Adapter 等任意 PEFT 方法。

**2-bit 优势显著**（Llama-2-7B WikiText-2 困惑度）：

| 方法 | 2-bit ppl | 3-bit ppl |
|---|---|---|
| QLoRA | 1.8e5（失效） | 1540（失效） |
| LoftQ | 1.0e3（失效） | 10.72 |
| ApiQ-bw | **7.59**（可用） | **5.77** |

**代码**：`github.com/BaohaoLiao/ApiQ`

---

### 9.5 LR-QAT（Qualcomm AI Research 2024，路线 B）

**来源**：arXiv:2406.06385，Qualcomm AI Research Amsterdam

**核心思想**：路线 B 的代表实现，将低秩适配器置于量化函数内部。NNCF LoRA QAT 的核心算法来源。

**核心公式**（Eq. 8）：

$$
\hat{W} = s \cdot \text{clip}\!\left(\!\left\lfloor \underbrace{\Phi_0}_{\text{冻结降精度 }W_0} + \frac{\alpha}{r} AB \right\rceil, -2^{b-1}, 2^{b-1}-1\right)
$$

- $\Phi_0 = \varphi(W_0 / s_0)$：降精度算子，将冻结的 W₀ 存为低精度格式  
- STE（Straight-Through Estimator）用于反向传播 ⌊·⌋ 和 clip(·)  
- 训练后：`W_Z = clip(⌊Φ₀ + αAB/r⌋)` 是单一低比特整数矩阵，推理零额外开销

**降精度算子 φ 关键选择**（3-bit 场景）：

| φ 类型 | Φ₀ 精度 | W₀ 内存占用 | 3-bit 收敛 |
|---|---|---|---|
| BF16 | 16-bit float | 2 bytes | 最好 |
| Q3.5（固定小数点） | 8-bit fixed | 1 byte | 等同 BF16 |
| INT-3（双打包） | 4-bit int（粗粒度） | 0.5 byte | **需 LoftQ 初始化才可用** |

**梯度检查点**：对 ΦAB 乘积使用 gradient checkpointing，避免前向中间结果占用大量显存。

**3-bit 性能对比**（LLaMA-2 7B，W3pc，WikiText-2 ppl↓）：

| 方法 | 7B ppl | 内存需求（训练） |
|---|---|---|
| RTN | 26.73 | - |
| GPTQ | 8.37 | ~7 GB |
| AWQ | 24.00 | ~7 GB |
| OmniQuant | 6.58 | ~16 GB |
| LR-QAT | **6.13** | **21 GB**（单卡可训） |
| FP16 基准 | 5.47 | - |

**代码**：`github.com/qualcomm-ai-research/LR-QAT`

---

### 9.6 NNCF LoRA QAT（Intel，路线 B）

**NNCF（Neural Network Compression Framework）** 是 Intel OpenVINO 生态的官方神经网络压缩框架。NNCF LoRA QAT 是 LR-QAT 算法路线在 NNCF 框架内的完整实现。

#### 核心公式（NNCF 特定实现）

```python
# NNCF LoRA QAT 前向（训练阶段）
W_q = FakeQuantize(W₀ + α/r · A @ B,
                   scale=s,          # 可学习量化 scale
                   zero_point=z,     # 可学习零点
                   bits=3)           # 目标 bit 宽

# 关键区别（vs LoftQ/QLoRA）：
# NNCF:         W_q = FQ(W₀ + AB)     ← AB 在量化函数 FQ 内部
# LoftQ/QLoRA:  W_q = FQ(W₀) + AB    ← AB 在量化函数 FQ 外部
```

#### 训练配置（3-bit 典型参数）

```python
nncf_config = {
    "target_bits": 3,                    # 目标 bit 宽
    "lora_rank": 64,                     # LoRA rank（影响精度/内存平衡）
    "alpha": 16,                         # LoRA 缩放因子
    "quantization_granularity": "per_channel",  # 或 per_group (128)
    "calibration_dataset": "alpaca",     # 128–512 样本
    "training_steps": 10000,
    "batch_size": 16,
    "learning_rate": 2e-4
}
```

#### 与其他方法的对比

| 维度 | SignRoundV2 (PTQ) | NNCF LoRA QAT (QAT) | LoftQ (PTQ+finetune) |
|---|---|---|---|
| 训练时间 | 数小时（校准） | GPU-天级 | 数小时 + 微调天级 |
| 推理 LoRA 开销 | 无 | **无**（融合） | 有 |
| 3-bit 精度上限 | 中等 | **最高** | 较高 |
| 所需数据 | 128 calibration | 大规模训练数据 | 128 calibration + 微调数据 |
| 适用场景 | 快速 PTQ 部署 | 最高精度，有训练资源 | 有微调需求 |

---

### 9.7 ParetoQ（NeurIPS 2025）

**来源**：arXiv:2502.02631，Meta AI Research，NeurIPS 2025

**核心思想**：首个统一框架严格比较 1/1.58/2/3/4-bit 量化性能，揭示不同 bit 宽在 size-accuracy Pareto 曲线上的位置。

**关键发现**：

1. **2-3 bit 相变（Learning Transition）**：
   - 3-bit 以上：量化模型分布接近预训练分布（类 fine-tuning）
   - 2-bit 以下：表示分布大幅偏移（需要类 pre-training 的大幅度更新）

2. **3-bit 最优方法 = LSQ**（Learned Step Size Quantization）：  
   对于 3-bit 对称格点（含 0 的格点）LSQ 更有效：

$$
\hat{W} = s \cdot \text{clip}\!\left(\!\left\lfloor \frac{W}{s} \right\rceil, -4, 3\right), \quad
\frac{\partial L}{\partial s} = \frac{1}{\sqrt{N \cdot 3}} \sum_i \partial_{\text{STE}}
$$

3. **Pareto 结论**：
   - 3-bit 与 2-bit 在 size-accuracy Pareto 上接近（2-bit 更省内存）
   - 4-bit 在 Pareto 曲线上被 3-bit **支配**（同精度下 3-bit 更小）
   - 二值量化（1-bit）在所有 Pareto 指标上被多比特量化支配

**600M ternary 模型超越 3B ternary SOTA**（仅用 1/5 参数）

---

### 9.8 3-bit SE（Scale Equalization）

**SE（Scale Equalization）** 是 NNCF 和 SmoothQuant 系中用于量化预处理的通道均衡化技术，在做 3-bit PTQ 之前消除权重通道间 scale 差异。

#### 数学基础

对线性层 $Y = XW$，引入对角缩放矩阵 $D$（吸收到相邻层）：

$$
Y = \underbrace{X \cdot D^{-1}}_{\text{缩放激活（吸收进前层）}} \cdot \underbrace{D \cdot W}_{\text{均衡后权重}}
$$

选择 $D$ 使 $DW$ 的各输入通道 max 值近似相等，降低 absmax 量化的通道不均匀性。

#### SmoothQuant（Xiao 等，ICML 2023）公式

$$
d_j = \frac{\max(|X_j|)^\alpha}{\max(|W_j|)^{1-\alpha}}, \quad \alpha \in [0, 1]
$$

- $\alpha = 0.5$：激活/权重均等分担均衡化
- $\alpha \to 1$：将量化难度尽量迁移到权重侧（保护激活通道精度）

#### NNCF 3-bit SE 实现流程

```python
# 步骤 1：收集权重通道 scale 统计
channel_scales = {}
for layer_name, layer in model.named_modules():
    if isinstance(layer, nn.Linear):
        # per-output-channel 最大绝对值
        max_per_channel = layer.weight.abs().max(dim=1).values
        channel_scales[layer_name] = max_per_channel

# 步骤 2：计算均衡化因子（折叠进上一层 LayerNorm/Linear）
equalization_factor = compute_equalization(channel_scales, alpha=0.5)

# 步骤 3：折叠 scale 到相邻层
for layer_name, factor in equalization_factor.items():
    layer.weight *= factor[:, None]      # 当前层权重乘以均衡因子
    prev_layer.weight /= factor[None, :] # 前层吸收补偿

# 步骤 4：3-bit 量化（均衡化后各通道 scale 相近，量化误差更小）
quantize_model(model, bits=3)
```

**3-bit 精度提升**：SE 单独可将 3-bit MMLU 提升 **0.5–1.5%**（依赖模型和通道分布）

**特点**：纯数据无关（data-free），无需校准数据，可与任何 PTQ 方法组合。

---

### 9.9 3-bit SE + AWQ（Scale Equalization + Activation-aware Weight Quantization）

**AWQ（Lin 等，arXiv 2306.00978）** 是 MIT EECS 提出的激活感知权重量化方法，识别"显著权重"并给予通道保护 scale。

#### AWQ 核心思想

LLM 中仅 ~1% 权重对精度至关重要——这些权重对应**高激活幅值**的输入通道。AWQ 通过 per-input-channel scale $s$ 保护这些通道：

$$
\text{argmin}_{s} \left\| Q\!\left(W \cdot \text{diag}(s)\right) \cdot \text{diag}(s)^{-1} - W \right\|
$$

等价地，相当于对"重要通道"用更大 scale 保护（减小其量化相对误差）：

$$
s_j \propto \max(|X_j|)^\alpha, \quad \alpha \in [0, 1]
$$

$s$ 通过校准数据集的激活统计确定，然后折叠进前一层（`LayerNorm`, `Linear`, `Embedding`）。

#### SE vs AWQ 的本质区别

| 维度 | SE（Scale Equalization） | AWQ |
|---|---|---|
| 统计量来源 | 权重通道 max 值 | **激活** max 值（需校准数据） |
| 均衡方向 | 沿输出通道（out-channel）均匀化 | 沿输入通道（in-channel）保护显著权重 |
| 目标 | 权重通道间 scale 均匀（减少量化难度） | 保护激活大的权重通道（减少显著误差） |
| 数据依赖 | 纯权重分析（data-free） | 需要 128 样本校准数据 |
| 互补性 | 解决权重内部不均匀 | 解决激活-权重耦合敏感性 |

#### 3-bit SE + AWQ 组合流程

```python
# 步骤 1：收集激活统计（per-input-channel max 激活幅值）
activation_stats = calibrate(model, calibration_data)  # 128 样本

# 步骤 2：Scale Equalization（SE）— 权重通道均匀化
equalization_scales = compute_se_scales(model.weights, alpha=0.5)
apply_se_folding(model, equalization_scales)

# 步骤 3：AWQ — 激活感知保护 scale（在 SE 均衡后的模型上）
awq_scales = compute_awq_scales(activation_stats, model.weights, alpha=0.5)
apply_awq_folding(model, awq_scales)

# 步骤 4：3-bit PTQ（GPTQ / RTN 作用于经双重预处理的权重）
quantize_model(model, bits=3, method="gptq")
```

#### 3-bit PTQ 性能对比（LLaMA-2 7B，WikiText-2 ppl↓）

| 方法 | 3-bit ppl |
|---|---|
| RTN 3-bit（无预处理） | 26.73 |
| SE 3-bit（NNCF Equalization） | ~8.5–10（估计） |
| GPTQ 3-bit | 8.37 |
| AWQ 3-bit | 6.24 |
| SE + AWQ 3-bit | ~6.0（估计） |
| OmniQuant 3-bit | 6.03 |
| LR-QAT 3-bit（QAT） | 6.13 |
| FP16 基准 | 5.47 |

**SE + AWQ 的互补性**：SE 消除权重通道 scale 不均，AWQ 保护激活幅值大的通道，两者各自解决不同维度的量化误差源，组合效果接近 OmniQuant 和 LR-QAT 水平。

---

### 9.10 SKILL.md 新增论文总览

| 论文 | 类型 | 关键创新 | 代码 |
|---|---|---|---|
| **LoftQ** (ICLR 2024) | PTQ + LoRA init | 迭代 SVD 从量化残差初始化 A/B | github.com/yxli2123/LoftQ |
| **PiSSA** (NeurIPS 2024) | PTQ + LoRA init | 主成分→适配器，残差→量化 | github.com/GraphPKU/PiSSA |
| **IR-QLoRA** (ICML 2024) | PTQ + ICQ + IEC | 信息熵最大化 τ 校准 + 弹性跳连 | github.com/htqin/ir-qlora |
| **ApiQ** (ICML 2024) | PTQ + 激活对齐 | 最小化激活误差（非权重误差），层序传播控制 | github.com/BaohaoLiao/ApiQ |
| **ParetoQ** (NeurIPS 2025) | QAT 统一框架 | 1-4bit Pareto 对比，3/4-bit 用 LSQ | NeurIPS 2025 |
| **LR-QAT** (2024) | QAT + 低秩（路线 B） | LoRA 置于 FQ 内部，训练后融合零推理开销 | github.com/qualcomm-ai-research/LR-QAT |

**方法精度-效率权衡总结**：

```
训练成本（低→高）：
  RTN < GPTQ < SE < AWQ ≈ SE+AWQ < LoftQ < ApiQ < LR-QAT/NNCF-QAT

3-bit 精度（低→高，LLaMA-7B ppl↓）：
  RTN(26.7) < GPTQ(8.4) < AWQ(6.2) ≈ SE+AWQ ≈ OmniQuant(6.0) < LR-QAT(6.1)
  
3-bit LoRA 微调精度（低→高，LLaMA-7B MMLU）：
  QLoRA(37.8%) < QA-LoRA(37.4%) < IR-QLoRA(38.4%) ← PTQ+LoRA 路线
  LR-QAT ≈ NNCF-QAT ← QAT 路线（精度更高，训练成本高）
```

---

## 十、GGUF 格式中的 3-bit 量化

> 参考：llama.cpp `ggml-quants.c` / `ggml-common.h`；Unsloth Qwen3.6-35B-A3B-GGUF

GGUF 是 llama.cpp 生态的通用模型格式，内置了从 1-bit 到 8-bit 的多种量化类型。其中与 3-bit 相关的格式分为两大类：**k-quants 标量量化**（`Q3_K` 系列）和 **i-quants 向量量化**（`IQ3_XXS`、`IQ3_S`）。以 Qwen3.6-35B-A3B 为例，Unsloth 发布了如下 3-bit 变体：

| 文件名 | 格式 | 文件大小 | bpw |
|---|---|---|---|
| `UD-Q3_K_S.gguf` | Q3_K_S | 15.4 GB | ~3.5 |
| `UD-Q3_K_M.gguf` | Q3_K_M | 16.6 GB | ~3.9（部分层 4-bit） |
| `UD-Q3_K_XL.gguf` | Q3_K_XL | 16.8 GB | ~3.9（imatrix 优化） |
| `UD-IQ3_XXS.gguf` | IQ3_XXS | 13.2 GB | 3.0625 |
| `UD-IQ3_S.gguf` | IQ3_S | 13.7 GB | 3.3125 |

> **注：** `Q3_K_M` / `Q3_K_XL` 采用混合精度策略，关键层（attention、embedding）使用更高 bit（Q4/Q5）以保持精度，整体平均约 3.9 bpw。

---

### 10.1 Q3_K — 经典 k-quants 3-bit 标量量化

**来源：** llama.cpp 原创，与 GPTQ 独立设计，不依赖 Hessian 信息。

#### 数据结构（`block_q3_K`，`QK_K = 256` 个权重/super-block）

```c
// ggml-common.h
#define QK_K 256   // super-block size (256 weights)

typedef struct {
    uint8_t  hmask[QK_K/8];   // 32 bytes: high bit (bit-2) for each weight
    uint8_t  qs[QK_K/4];      // 64 bytes: low 2 bits packed, 4 weights/byte
    int8_t   scales[12];      // 12 bytes: 6-bit sub-block scales, 16 sub-blocks of 16
    ggml_fp16_t d;            // 2  bytes: super-block scale (fp16)
} block_q3_K;
// 总计: 2 + 12 + 64 + 32 = 110 bytes / 256 weights = 3.4375 bits/weight
```

#### 编码方案

每个权重 w 的范围为 `[-4, +3]`（有符号 3-bit，偏移为 4）：

```
存储结构（per weight）：
  bit[1:0] → qs[k/4] >> ((k%4)*2)  & 0x3   (低 2 bits，无符号)
  bit[2]   → hmask[k/8] >> (k%8)   & 0x1   (高 1 bit)
  3-bit 值  = low2bits | (high_bit << 2)     (无符号 0–7)
  真实量化值 = (3-bit 值) - 4               (有符号 -4 到 +3)

sub-block scale（16 个权重为一组，共 16 sub-blocks per super-block）：
  6-bit 整数，打包在 scales[12] 中（复杂位交织）
  真实 scale = d * sub_scale（其中 d 为 fp16 super-block scale）
```

#### 解压公式

```c
// dequantize_row_q3_K（ggml-quants.c 简化版）
for each sub-block (16 elements):
    dl = d_all * (scales[is] - 32)          // 6-bit → [-32, +31]，再乘 super-block d
    for l in 0..15:
        // 取低 2 bits
        q2 = (qs[l] >> shift) & 0x3
        // 取高 1 bit（来自 hmask）
        high = (hmask[l] & m) ? 0 : 4       // m 是 bit mask，按子块循环
        y[l] = dl * (q2 - high)             // 真实浮点值
```

#### Q3_K 变体

| 变体 | 策略 | 典型 bpw |
|---|---|---|
| `Q3_K_S` | 所有层统一 Q3_K | ~3.5 |
| `Q3_K_M` | 关键层（attention、FFN gate）用 Q4_K，其余 Q3_K | ~3.9 |
| `Q3_K_XL` | Q3_K_M + imatrix（importance matrix 加权量化） | ~3.9 |

> **imatrix（importance matrix）**：通过少量校准数据前向传播，统计每个权重通道的激活幅度，优先保留重要通道的精度。由 llama.cpp 的 `llama-imatrix` 工具生成，Unsloth 提供了预计算的 `imatrix_unsloth.gguf_file`。

---

### 10.2 IQ3_XXS — 3.0625 bpw（D4 lattice 向量量化）

**核心思路：** 使用 256-entry 4D codebook（D4 lattice 子集），每组 4 个权重查一次表，附加 7-bit 符号翻转以扩展等效码本容量至 256×128 = 32768 个向量。

#### 数据结构（`block_iq3_xxs`，每 `QK_K=256` 个权重一块）

```c
typedef struct {
    ggml_fp16_t d;                    // 2  bytes: super-block scale
    uint8_t     qs[3*QK_K/8];        // 96 bytes: 8-bit indices (每 4 weights 一个 uint8)
    // 最后 4*(QK_K/32) = 32 bytes: scales_and_signs
    //   高 28 bits: 7-bit sign mask × 4 sub-groups per 32-element block
    //   低  4 bits: 4-bit sub-block scale
} block_iq3_xxs;
// 总计: 2 + 96 + 32 = 130 bytes 有效 / 256 weights ≈ 3.0625 bits/weight（含 overhead）
```

#### 编码方案

```
codebook kgrid_256[256]：256 个 4D 向量，每维取奇数值 {1,3,5,7,9,11,13,15}（8选1，3 bits/维）
  → 每个码字代表 4 个范围在 [-8, +8] 的奇数权重值

每 32 个权重的存储：
  qs[0..7]: 8 个 uint8，每个是 codebook 中的 8-bit 索引（对应 4D vector）
  scales_and_signs[ib]:
    bits[31:28] = 4-bit sub-block scale (0–15，经 d 缩放)
    bits[27:21] = 7-bit sign mask（第 0 个 4D group 的 7 个符号翻转）
    bits[20:14] = 7-bit sign mask（第 1 个 4D group）
    ... （4 个 7-bit sign masks）

解压一个 4D group（4 个权重）：
    grid_vec = kgrid_256[qs[2*l+k]]   // 4D FP32 向量 {1,3,5,...,15}
    signs    = scales_and_signs_byte
    for j in 0..3:
        y[j] = db * grid_vec[j] * (signs & kmask[j] ? -1 : +1)
    // db = d * (0.5 + sub_scale) * 0.5
```

**精度损失分析：**
- 每维只能取 8 个奇数值，等效离散程度低于标量 INT4（16 个值）
- 通过 Hessian 加权的 imatrix 优化可补偿部分损失
- 实测（Llama-3-8B，WikiText-2 ppl）：Q3_K_S（5.60）< IQ3_XXS（5.62）< IQ3_S（5.51）

---

### 10.3 IQ3_S — 3.3125 bpw（512-entry D4 lattice，更高精度）

**相比 IQ3_XXS 的改进：**
- codebook 扩大至 **512 entries**（9-bit index，vs XXS 的 8-bit）
- sign mask 扩展至 **8 bits**（vs XXS 的 7 bits）
- 每组 4 个权重提升了码本分辨率，量化误差更小

#### 数据结构（`block_iq3_s`）

```c
typedef struct {
    ggml_fp16_t d;             // 2  bytes: super-block scale
    uint8_t     qs[QK_K/4];   // 64 bytes: 低 8 bits of 9-bit index（每 4 weights 一个）
    uint8_t     qh[QK_K/32];  // 8  bytes: 高 1 bit of 9-bit index（每 8 entries 一个 byte）
    uint8_t     signs[QK_K/8];// 32 bytes: 8-bit sign mask per 8 weights（2 个 4D groups）
    uint8_t     scales[QK_K/32];// 8 bytes: 4-bit sub-block scale pairs
} block_iq3_s;
// 总计: 2 + 64 + 8 + 32 + 8 = 114 bytes / 256 weights = 3.5625 bits/weight（含 overhead）
// 注：实际 bpw 标注为 3.3125，是对纯权重 bits 的统计（不含 scale/metadata 比例）
```

#### 解压公式

```c
// dequantize_row_iq3_s（ggml-quants.c 简化）
for ib32 in 0 .. QK_K/32:
    db1 = d * (1 + 2*(scales[ib32/2] & 0xf))   // 奇数 scale 值 {1,3,5,...,31}
    db2 = d * (1 + 2*(scales[ib32/2] >> 4))

    for l in 0..3:  // 4 个 4D groups
        // 9-bit index = qs[2*l] | ((qh >> (2*l)) & 1) << 8
        idx = qs[2*l+0] | ((qh[0] << (8-2*l)) & 256)
        grid1 = iq3s_grid[idx]   // 512-entry codebook，4D vector

        idx2 = qs[2*l+1] | ((qh[0] << (7-2*l)) & 256)
        grid2 = iq3s_grid[idx2]

        for j in 0..3:
            y[j+0] = db1 * grid1[j] * (signs[l] & mask[j+0] ? -1 : +1)
            y[j+4] = db1 * grid2[j] * (signs[l] & mask[j+4] ? -1 : +1)
```

---

### 10.4 TQ1_0 / TQ2_0 — GGUF 内的三值量化（BitNet 专用）

llama.cpp 还为 BitNet 类模型增加了两种三值量化格式（来源：`ggml-quants.c`），但 Qwen3.6 GGUF 中未包含这两种格式（仅适用于 BitNet/TriLM 系列）：

**TQ1_0（~1.69 bpw）：**
- 以 3 进制（base-3）打包，5 个三值权重打包入 1 个 uint8（3^5 = 243 < 256）
- 解压使用乘法 `q * pow3[n]`，取高 2 bits 得到 `{0,1,2}` 再映射 `→ {-1,0,+1}`

**TQ2_0（~2.0 bpw）：**
- 以 2 bits 打包，4 个三值权重/byte，等同于 GGML 内部对称 ternary
- 解压：`(qs[j] >> (l*2)) & 3 - 1` → `{-1, 0, +1}`

---

### 10.5 GGUF 3-bit 格式对比总结

| 格式 | bpw | block 结构 | codebook | 解压复杂度 | 适用场景 |
|---|---|---|---|---|---|
| **Q3_K_S** | 3.4375 | 256-weight super-block，2+1 bit 标量 | 无 | 极低：shift+AND+mul | 通用，最快 |
| **Q3_K_M** | ~3.9 | 混合精度（关键层 Q4_K） | 无 | 极低 | 均衡精度/速度 |
| **Q3_K_XL** | ~3.9 | Q3_K_M + imatrix | 无 | 极低 | 最佳 Q3 精度 |
| **IQ3_XXS** | 3.0625 | 32-weight block，256-entry 4D LUT | 256×4D（1KiB） | 低：LUT + sign XOR | 最小体积 |
| **IQ3_S** | 3.3125 | 32-weight block，512-entry 4D LUT | 512×4D（2KiB） | 低：LUT + sign XOR | 精度/体积平衡 |
| **TQ1_0** | 1.69 | 256-weight，base-3 packed | 无 | 中：mod + shift | BitNet 专用 |
| **TQ2_0** | 2.0 | 256-weight，2bit ternary | 无 | 极低：2-bit unpack | BitNet 专用 |

#### 精度对比（Llama-3-8B，WikiText-2 ppl ↓）

| 格式 | ppl |
|---|---|
| FP16 基准 | 6.24 |
| Q4_K_M | 6.57 |
| Q3_K_S | ~7.0 |
| Q3_K_M | ~6.7 |
| IQ3_S | ~6.8 |
| IQ3_XXS | ~7.2 |
| IQ2_M | ~8.0 |

> 数据来源：llama.cpp wiki / Unsloth Dynamic 2.0 benchmarks

#### OpenVINO 适配建议

| 格式 | OpenVINO 现状 | 适配难度 | 优先级 |
|---|---|---|---|
| Q3_K_S | ❌ 未支持 | 低（与 GPTQ W3 同原理） | ★★★★ 高 |
| Q3_K_M | ❌ 未支持 | 低（混合精度 kernel） | ★★★☆ 中高 |
| IQ3_S | ❌ 未支持 | 中（2KiB L1 codebook） | ★★★☆ 中高 |
| IQ3_XXS | ❌ 未支持 | 中（1KiB L1 codebook） | ★★☆☆ 中 |
| TQ1_0/TQ2_0 | ❌ 未支持 | 中（BitNet 专用路径） | ★☆☆☆ 低（需 BitNet 模型） |

**Q3_K 适配路径（最高优先）：** 直接扩展现有 OpenVINO INT4/INT8 解压 kernel，修改 bit-width 和 mask 参数即可复用 `fully_connected_kernel` 框架。IQ3_S/IQ3_XXS 需要额外将 2KiB/1KiB codebook 放入 OpenCL `__constant` 内存。

