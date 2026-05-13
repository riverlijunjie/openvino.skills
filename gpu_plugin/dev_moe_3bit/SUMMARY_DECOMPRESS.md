# 3-bit 权重压缩算法：推理解压细节分析

> 补充文档：从 GPU Kernel 实现角度分析 [SUMMARY.md](./SUMMARY.md) 中所有算法在推理时的解压流程、内存布局和内核设计考量。
> Jira: CVS-180191 / CVS-182951 — OpenVINO MoE GPU 3-bit weight support

---

## 目录

1. [解压范式分类](#一解压范式分类)
2. [标量量化族](#二标量量化族)
   - BitNet b1.58
   - GPTQ / RTN W3/W4
   - SignRoundV2 W2/W3
   - Scale Equalization 系列
   - ParetoQ
   - LR-QAT / NNCF
3. [PTQ + LoRA 族](#三ptq--lora-族)
   - LoftQ / PiSSA / IR-QLoRA / ApiQ
4. [向量量化族](#四向量量化族)
   - QuIP#
   - AQLM
5. [格码计算型](#五格码计算型)
   - QTIP / YAQA
6. [内存布局汇总对比表](#六内存布局汇总对比表)
7. [解压流程伪代码](#七解压流程伪代码)
8. [OpenVINO 实现建议](#八openvino-实现建议)
9. [GGUF 3-bit 格式解压细节](#九gguf-3-bit-格式解压细节)
   - Q3_K 系列（标量 k-quants）
   - IQ3_XXS（256-entry D4 lattice）
   - IQ3_S（512-entry D4 lattice）
   - TQ1_0 / TQ2_0（三值 BitNet 专用）

---

## 一、解压范式分类

推理时解压的核心目标是在不将完整解压后的 FP16/BF16 权重存储到 HBM 的情况下，以最低延迟完成 GEMM 操作。根据解压机制，所有算法可分为三大范式：

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          推理解压范式分类                                          │
│                                                                                 │
│  范式 A：标量量化       范式 B：向量量化         范式 C：格码计算                    │
│  ──────────────        ──────────────          ──────────────                  │
│  每个权重 = 独立标量     每 g 个权重 = 一个        每个权重 = 从 L 位流                │
│  (INT2/3/4)             码字 (codebook lookup)   解码（算法计算无表）                │
│                                                                                 │
│  BitNet b1.58           QuIP# E8P               QTIP 1MAD/3INST                │
│  GPTQ / RTN             AQLM 2×8                QTIP HYB                       │
│  SignRoundV2            AQLM 1×16 (L1 miss!)                                  │
│  SE / SE+AWQ            IQ3_XXS (1KiB D4 LUT)                                 │
│  LoftQ/PiSSA/IR-QLoRA  IQ3_S   (2KiB D4 LUT)                                 │
│  LR-QAT / NNCF                                 格码计算（无表）                  │
│  ParetoQ                                                                        │
│  Q3_K（GGUF k-quants）                                                          │
│  TQ1_0/TQ2_0（GGUF ternary）                                                   │
│                                                                                 │
│  特征：                  特征：                   特征：                           │
│  - 零 LUT               - 需 codebook (L1 fit?)  - 零或极小 LUT                 │
│  - 线性计算              - 查表并求和              - 纯 ALU 计算                    │
│  - ALU 瓶颈             - BW 或 cache 瓶颈        - LCG / hash 伪随机              │
│  - 2–5 instr/weight     - 2–5 instr/weight       - 2–4 instr/weight             │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**Roofline 考量：** 解压速度要 ≥ 内存带宽速率，否则将从 memory-bound 变成 compute-bound。目标：所有范式 ≤ 5 指令/权重，使 SM 在 ldg（HBM load）完成时已完成解码。

---

## 二、标量量化族

### 2.1 BitNet b1.58

**权重类型：** 三值 `{-1, 0, +1}`，每权重 1.585 bit

#### 编码格式（推理时存储）

```
打包方式（GPU 变体）：
  4 个 ternary 值 → 1 个 int8
  编码：{-1→0b00, 0→0b01, +1→0b10}（2 bits each）

  packed[byte_idx] = (w[0] & 0x3) | ((w[1] & 0x3) << 2)
                   | ((w[2] & 0x3) << 4) | ((w[3] & 0x3) << 6)
```

#### GPU Kernel 解压流程

```
HBM (packed int8)
      │
      ▼ ldg（128B cache line = 128 个三值权重）
SRAM (shared memory / register file)
      │
      ▼ 解包（4 ops per int8）：
        w0 = (packed >> 0) & 0x3  → 映射: 0→-1, 1→0, 2→+1
        w1 = (packed >> 2) & 0x3
        w2 = (packed >> 4) & 0x3
        w3 = (packed >> 6) & 0x3
      │
      ▼ W1.58 A8 matmul（激活 INT8，权重 {-1,0,+1}）
        y += w[k] * x[k]  = MAX(x, -x, 0) 选择
```

#### 内存布局

```
qweight: (K // 4, N) int8    ← 4 ternary/byte，行优先
scale_w: (1,) fp32            ← 全局 per-tensor scale（论文设计）
scale_a: (B, T, N) int8      ← 激活 per-token scale（动态量化）
```

- **HBM → SM 传输量：** 原始 FP16 的 8 分之一（2 bit/weight vs 16 bit）
- **L1 friendliness：** ✅ 无 codebook，纯 register 操作
- **CPU 路径（bitnet.cpp）：** 利用 TL/BF16 扩展指令，2 个 uint8 = 8 三值权重，使用 bit manipulation 展开

---

### 2.2 GPTQ / RTN

#### 2.2.1 W4（最常见）

```
内存布局：
  qweight: (K // 8, N) int32       ← 8 个 INT4/int32，column-major
  scales:  (K // group_size, N) fp16
  qzeros:  (K // group_size, N // 8) int32   ← 8 个 zero_point/int32

解包（per-column, 8 weights per int32）：
  for i in 0..7:
    w_int = (qweight[k//8, n] >> (i * 4)) & 0xF   ← shift + AND
    zp    = (qzeros[k//gs, n//8] >> ((n%8) * 4)) & 0xF
    scale = scales[k//gs, n]
    w_fp  = (w_int - zp) * scale                    ← sub + mul
```

**指令数：** 4（shift, AND, sub, mul-fp16）

#### 2.2.2 W3（GPTQ 3-bit）

W3 打包较复杂，ExLlama/AutoGPTQ 常用两种方案：

**方案一（10 weights per int32，浪费 2 bits）：**
```
qweight: (K // 10, N) int32   ← 30 bits used, 2 wasted per int32
w_int = (qweight[k//10, n] >> ((k%10) * 3)) & 0x7
```

**方案二（perfect 3-bit packing，32 weights per 3 int32s）：**
```
qweight: (K // 32 * 3, N) int32   ← 32 weights in 3 consecutive int32s
# 从 bit 偏移 [0,3,6,9,12,15,18,21,24,27,30, 33 跨int32, ...]
i_int32 = (k * 3) // 32
i_bit   = (k * 3) % 32
if i_bit <= 29:
    w_int = (qweight[i_int32, n] >> i_bit) & 0x7
else:  # 跨越 int32 边界
    lo = (qweight[i_int32, n] >> i_bit)
    hi = (qweight[i_int32+1, n] << (32 - i_bit))
    w_int = (lo | hi) & 0x7
```

**指令数：** 5–6（处理跨 int32 边界时需额外 shift+OR）

---

### 2.3 SignRoundV2 (W2/INT2)

> 详情见 SUMMARY.md §八

#### 内存布局

```
qweight: (K // 16, N) int32   ← 16 个 INT2/int32
scales:  (K // 128, N) fp16   ← group_size = 128
qzeros:  (K // 128, N // 16) int32  ← 16 个 zp/int32
```

#### 解包公式

```
k_tile  = k // 16
k_mod   = k % 16
w_int   = (qweight[k_tile, n] >> (k_mod * 2)) & 0x3     ← shift 0/2/4/.../30, AND 0x3
zp      = (qzeros[k//128, n//16] >> ((n%16)*2)) & 0x3
scale   = scales[k//128, n]
w_fp    = (w_int - zp) * scale
```

**指令数：** 4（同 W4）

#### Triton Kernel 结构（来自论文开源实现）

```python
# 按 block 加载 packed qweight
qw_block = tl.load(qweight_ptr + ...)                    # int32[BLOCK_K//16, BLOCK_N]
# 展开为 16 行
for shift in [0, 2, 4, ..., 30]:
    w_2bit = (qw_block >> shift) & 0x3                   # int32 → INT2
    w_fp   = (w_2bit.to(fp16) - zp_fp) * scale          # dequant
    acc   += w_fp * x_block                              # accumulate
```

**关键优化：** 每个 thread 处理 16 个连续 k 的 INT2 → 避免跨 thread 的 scatter，寄存器压力低。

---

### 2.4 Scale Equalization (SE) / SE+AWQ

SE 只是预处理技巧，**推理时布局与 GPTQ 相同**：

```
# 3bit SE：
qweight: (K // 10, N) int32   ← INT3，同 GPTQ W3
scales:  (K // group_size, N) fp16
# 解压：与 GPTQ W3 完全相同

# SE+AWQ（吸收 per-channel scale）：
# s_j ∝ (mean|x_j|)^α 被合并进 group scale
# 推理时 scale_merged = scale_group * per_channel_scale
# 解压：(w_int - zp) * scale_merged  ← 同一步完成，零额外开销
# 或者：(w_int - zp) * scale_group * per_channel_scale ← 2 次 mul，但可预融合
```

---

### 2.5 ParetoQ

QAT 方案，学习步长 s，推理时**等同于 INT3/INT4 标准解压**：

```
量化权重存储：同 GPTQ
推理时：w_fp = w_int * s   # s 即量化步长（group 或 per-channel）
        # 没有 zero_point（对称量化），所以是 3 ops（shift+AND+mul）
```

---

### 2.6 LR-QAT / NNCF LoRA QAT

**关键点：推理时无 LoRA 开销**，LoRA 矩阵 B·A 在部署前已融合（absorbed）进量化权重：

```
训练时：y = dequant(W_Z) · x + (α/r) · B · (A · x)
推理时：y = dequant(W_Z') · x    # W_Z' = repost-quantize(W + α/r·BA)
```

布局与 GPTQ W3/W4 完全相同，无额外内存开销。

---

## 三、PTQ + LoRA 族

### 3.1 LoftQ / PiSSA / IR-QLoRA / ApiQ

这一族的核心差别在于：**推理时必须保留并应用 LoRA 分支**（LoRA 不融合）。

#### 内存布局

```
W_q: (K // 8, N) int32     ← INT4 NF4（同 QLoRA/bitsandbytes 格式）
A:   (K, r)     fp16/bf16  ← Low-rank factor A（"up projection"）
B:   (r, N)     fp16/bf16  ← Low-rank factor B（"down projection"）
scales_q: (K // group_size, N) fp16
```

#### 推理计算流程

```
输入 x: (batch, K)

步骤 1：LoRA 分支（FP16 matmul）
  mid = x @ B.T    ← (batch, r)   --- r 通常 8–64
  lora_out = mid @ A.T  ← (batch, K)

步骤 2：量化分支（dequant + matmul）
  W_fp = dequant(W_q, scales_q)  ← on-the-fly（不物化到 HBM）
  main_out = x @ W_fp            ← (batch, N)

步骤 3：合并
  y = main_out + (α / r) * lora_out
```

#### 计算开销分析

```
FLOPs overhead（per token）：
  主路径：2 × K × N  FP16 MACs（对比量化后 ≈ 2×K×N/4 实际计算）
  LoRA 路径：2 × K × r + 2 × r × N = 2r(K+N) FP16 MACs

当 r=16, K=N=4096：
  LoRA overhead ≈ 2×16×8192 / (2×4096×4096) ≈ 0.78%  → 可忽略

内存 overhead：
  |A|+|B| = r×(K+N)×2 bytes = 16×8192×2 = 256KiB per layer
  相对 W_q = K×N/2 bytes = 8MiB → LoRA 约 3%
```

**LoRA 是否需要单独 Kernel：**
- 大 batch（prefill）：LoRA 可融入 INT4 GEMM 后的 FP16 残差 GEMM
- 小 batch（decode）：两次独立 FP16 GEMV，overhead 显著（但 r 小）

---

## 四、向量量化族

### 4.1 QuIP# (E8P codebook)

QuIP# 是目前向量量化中 L1-efficient 的代表作。

#### 核心思想：Randomized Hadamard Transform（RHT）

```
量化时（训练后，存储前）：
  Ŵ = D_U · W · D_V     # D_U, D_V 是随机对角符号矩阵
  再向量量化：Ŵ → codebook indices

推理时（恢复）：
  x_hat = D_V · x        # 激活也经过 V-transform
  y_hat = Ŵ_decoded · x_hat
  y     = D_U^T · y_hat  # 输出经过 U^T-transform
```

RHT 的作用：将权重分布接近球面均匀分布，使 E8-lattice codebook 最优。

#### E8P Codebook 编码格式

```
每个 codeword（编码 8 个 FP16 权重）：
  16 bits total = 8 bits (index into S) + 7 bits (sign flips) + 1 bit (±1/4 shift)

  S: 256 × 8 = 2KiB base codebook（FP16，存入 L1/常量内存）
  完整逻辑码本：2^16 entries，但按需解码，非显式存储

codeword 解码：
  base_vec = S[codeword[15:8]]         # 8-FP16 向量，1 次 L1 查找
  sign_mask = codeword[7:1]             # 7 bits
  shift_bit = codeword[0]
  for i in 0..7:
    base_vec[i] *= (sign_mask >> i) & 1 ? -1 : 1   # 7 次 XOR（可 SIMD）
  # E8 parity 隐含在第 8 个符号中（不单独存储）
  if shift_bit:
    base_vec += 0.25                    # 可选平移
```

#### 内存布局

```
indices:   (K // 8, N // 8, 2) uint8   ← 每 8 权重 = 1 个 16-bit codeword（分 2 字节）
           或 (K // 8, N // 8) uint16  ← 更直接
scales:    (K // group_size, N) fp16   ← group scale（可选，QuIP# 默认无 group scale）
sign_U:    (d_model // 8,) uint8       ← S_U per-entry signs（packed bits）
sign_V:    (d_model // 8,) uint8       ← S_V per-entry signs（packed bits）
S_base:    (256, 8) fp16               ← 1KiB，常量内存（编译期固定）
```

#### CUDA Kernel 结构（decode_matvec_e8p_kernel）

```
// 每个 warp 处理 1 个 output channel
// 每个 thread 处理 8 个连续 input channels（1 个 E8P codeword）

for k_tile in range(K // 8):
    cw  = indices[k_tile, n]             // 1 个 uint16，1 次 LDG
    base_idx   = cw >> 8                 // 8 bits → index into S
    sign_bits  = (cw >> 1) & 0x7F       // 7 bits
    shift_flag = cw & 1

    vec = S_base[base_idx]              // 1KiB L1 lookup（8 FP16）
    vec ^= sign_bits_to_mask(sign_bits) // SIMD XOR（等价 1 指令）
    vec += shift_flag * 0.25            // conditional add

    acc += dot(vec, x[k_tile*8 : k_tile*8+8])   // 8-wide FP16 dot

// 最终应用 S_U sign（RHT U 变换）
acc *= sign_U_val
```

**指令数：** < 5 per weight
**Codebook L1 cost：**
- S: 256 × 8 × 2B = 4KiB（原始）
- × 32（bank conflict 展开）= 128KiB 极端情况，但实际优化后约 4–8KiB
- H100 L1: 256KB → ✅ 轻松 fit

**性能数据（RTX 4090）：**
- QuIP# 2-bit：>50% peak memory BW，接近 W4 GPTQ 速度
- QuIP# 3-bit：略慢（3-bit codeword 设计略复杂）

---

### 4.2 AQLM

#### 核心架构：多码本加法

```
W[i, j*g : (j+1)*g] = Σ_{m=1}^{M} C_m[ b_{ij,m} ]
                       ↑ M 个码本 C_m 的对应 entry 之和
g = codebook 粒度（8 或 16）
M = 码本数量（1 或 2）
```

#### 关键配置对比

```
配置          g    M    bits/w    codebook size    内存
───────────────────────────────────────────────────────────────────
1×16 (1B16)  16   1    16-bit    2^16 × 16 FP16  = 2MiB   ← L1 miss!
2×8 (2B8)     8   2     8-bit    2× 256 × 8 FP16 = 8KiB   ← L1 ✅
───────────────────────────────────────────────────────────────────
```

**1×16 的问题：** 2MiB codebook >> L1 cache（H100=256KB, A100=192KB），每次 codebook 查找都 cache miss → 推理速度**慢于 FP16**！

#### 内存布局

```
# 2×8 配置（推荐）：
qcodes:     (d_out, d_in // 8, 2) uint8    ← 每 8 权重 = 2 个 uint8 code
codebooks:  (2, 256, 8) fp16               ← 8KiB，fit L1
scales:     (d_out,) fp16                  ← per-output-channel scale

# 1×16 配置（慢，仅作对比）：
qcodes:     (d_out, d_in // 16) uint16     ← 每 16 权重 = 1 个 uint16
codebooks:  (1, 65536, 16) fp16            ← 2MiB，L1 miss
```

#### CUDA Kernel（2×8 版本）

```
// Thread 负责 1 个 output channel，处理 K/8 个 code 块
for k8 in range(K // 8):
    b1 = qcodes[n, k8, 0]              // uint8，1 次 LDG
    b2 = qcodes[n, k8, 1]              // uint8，1 次 LDG

    vec1 = codebooks[0, b1, :]         // 8 FP16，L1 lookup
    vec2 = codebooks[1, b2, :]         // 8 FP16，L1 lookup
    vec  = vec1 + vec2                  // 8-wide FP16 add

    acc += dot(vec, x[k8*8 : k8*8+8])  // 8-wide dot
acc *= scales[n]
```

**指令数（2×8）：** ~4–5 per weight（2 LUT + 1 add + 1 dot）
**性能数据（H100, 1×16 vs 2×8）：**
- 1×16：81.5 tok/s（比 FP16 慢）
- 2×8：比 FP16 快 30%（L1 hit 带来质变）

---

## 五、格码计算型

### 5.1 QTIP / YAQA

QTIP 的核心洞察：**用 ALU 运算替换 LUT 查找**，实现"零 codebook"的向量量化解压。

#### 理论基础：Bitshift Trellis

```
格码参数：L=16（窗口宽度），k=2（bits per weight），V=1 或 2（每窗口权重数）

每个权重的解码输入：连续 L=16 bits 的 trellis 状态
比特位移性质：相邻 V 个权重的窗口只需 shift V bits → 完全并行解码
```

#### 三种 Code 的解压指令分析

**1MAD Code（4 instructions/weight）：**

```
// 输入：packed_bits（16-bit trellis 状态）
// LCG 步骤（1 MAD + 1 AND）：
x = (34038481 * packed_bits + 76625530) & 0xFFFFFFFF

// vabsdiff4（1 条 PTX 指令）：将 x 解释为 4 个 uint8，求绝对差之和
x = abs(x[0:8] - x[8:16]) + abs(x[16:24] - x[24:32])  // vabsdiff4
// 注意：上行是简化示意，实际是 PTX vabsdiff4.u32.u32.u32.add

// 归一化（1 MAD）：
w = scale * x + shift   // 映射到目标分布范围
```

```
PTX 示意：
mad.lo.u32  x, 34038481, packed, 76625530;    // LCG: 1 instr
vabsdiff4.u32.u32.u32.add  x, x, 0, 0;        // 4 bytes sum: 1 instr
fma.rn.f16  w, scale_f16, x_f16, shift_f16;   // normalize: 1 instr
// 共 3 PTX，但 mad 内部可拆为 2 MUL+ADD → 约 4 机器指令
```

**3INST Code（3 instructions/weight）：**

```
// 常量：a=89226354, b=64248484, m=0.922（magic FP16 常量）
// LCG（1 MAD）：
x = (89226354 * packed_bits + 64248484) & 0xFFFFFFFF

// FP16 bit manipulation（1 lop3 = 3-input logic）：
// 将 x 解释为 2 个 fp16，与 magic 常量异或
m1 = (x & 0xFFFF) ^ magic_fp16       // 低 16 位 XOR
m2 = (x >> 16) ^ magic_fp16          // 高 16 位 XOR
// lop3 可同时做：a ^ (b & c) | (d & ~e) 等 → 1 指令完成两次 XOR+mask

// FP16 加法（1 FADD）：
w = m1 + m2                           // 结果已在 fp16 范围

// 整体：3 条 ALU 指令，无内存访问！
```

**HYB Code（~2 instructions/weight，2KiB LUT）：**

```
// 哈希（1 MAD）：
x = x * x + x    // mod 2^32（自乘 + 自加，近似 Marsaglia-type hash）

// LUT 查找（bitshift + mask + 1 memory op）：
Q = 9  // LUT size parameter
idx  = (x >> (15 - Q)) & (2^Q - 1)     // 9-bit index
vec2 = LUT[idx]                          // 2D fp16 vector (2 entries)
                                         // LUT = 512×2×fp16 = 2KiB → L1 ✅

// 符号翻转（1 lop3）：
w1 = vec2[0] ^ sign_bit(x)
w2 = vec2[1] ^ sign_bit(x >> 1)

// 每权重 2 ALU + 1 LUT = ~2 amortized（V=2 时两权重共享 hash）
```

#### 内存布局

```
packed_trellis: (K * k // 8, N) uint8    ← k bits/weight，bit-packed，column-major
                                           k=2: 1 byte per 4 weights
                                           k=3: 3 bytes per 8 weights (complex alignment)
scales:         () fp32                  ← 单一全局 scale（per-layer）或 per-column
sign_U:         (K // 8,) uint8          ← RHT S_U 符号向量（packed bits）
sign_V:         (N // 8,) uint8          ← RHT S_V 符号向量（packed bits）
LUT_HYB:        (512, 2) fp16            ← HYB 专用，2KiB，烧入常量内存
```

#### CUDA Kernel 结构（QTIP 1MAD/3INST）

```
// 每 warp 处理 1 output，每 thread 处理 L=16 个连续 trellis bits

for k_tile in 0 .. K/16:
    // 加载 16 bits（2 bytes）
    trellis_bits = load_u16(packed_trellis + k_tile * 2)

    // 解压 V=1 个权重（3INST 版本）：
    x  = mad_lo(89226354, trellis_bits, 64248484)
    m1 = lop3(x, 0xFFFF, magic_fp16)   // XOR 低半部
    m2 = lop3(x >> 16, 0xFFFF, magic_fp16)
    w  = fp16_add(m1, m2)

    // 激活加权求和：
    acc += w * x_input[k_tile]

// 应用 sign_U（RHT U-transform 一步完成）
acc *= sign_U[output_row_idx]
```

**性能数据（RTX6000 Ada）：**
- QTIP 2-bit 1MAD：188 tok/s
- QuIP# 2-bit：186 tok/s（几乎持平）
- AQLM 2-bit（1×16）：81.5 tok/s（差 2.3×，因 codebook cache miss）

---

## 六、内存布局汇总对比表

| 算法 | 精度 | qweight 格式 | qweight Shape | scale 格式 | qzero | codebook | L1-safe | LoRA 推理 |
|------|------|-------------|---------------|-----------|-------|----------|---------|----------|
| **BitNet b1.58** | 2-bit ternary | 4×ternary/int8 | (K//4, N) int8 | (1,) fp32 | — | — | ✅ | ❌ |
| **GPTQ W4** | INT4 | 8×INT4/int32 | (K//8, N) int32 | (K//gs, N) fp16 | (K//gs, N//8) int32 | — | ✅ | ❌ |
| **GPTQ W3** | INT3 | 10×INT3/int32 | (K//10, N) int32 | (K//gs, N) fp16 | (K//gs, N//10) int32 | — | ✅ | ❌ |
| **SignRoundV2 W2** | INT2 | 16×INT2/int32 | (K//16, N) int32 | (K//128, N) fp16 | (K//128, N//16) int32 | — | ✅ | ❌ |
| **SE / SE+AWQ** | INT3/INT4 | 同 GPTQ | 同 GPTQ | fp16（融合 AWQ s） | 同 GPTQ | — | ✅ | ❌ |
| **ParetoQ** | INT3/INT4 | 同 GPTQ | 同 GPTQ | (K//gs, N) fp16 | 同 GPTQ（或无） | — | ✅ | ❌ |
| **LR-QAT/NNCF** | INT3/INT4 | 同 GPTQ | 同 GPTQ | (K//gs, N) fp16 | 同 GPTQ | — | ✅ | ❌ |
| **LoftQ/PiSSA** | INT4+LoRA | 同 GPTQ W4 | (K//8, N) int32 | fp16 | 同 GPTQ | — | ✅ | **✅ (r×(K+N))** |
| **QuIP# E8P** | ~2-bit | 16-bit codewords | (K//8, N//8) uint16 | (K//gs, N) fp16 | — | S: 256×8 fp16 = 4KiB | ✅ | ❌ |
| **AQLM 2×8** | ~2-bit | uint8 indices ×2 | (d_out, K//8, 2) uint8 | (d_out,) fp16 | — | (2, 256, 8) fp16 = 8KiB | ✅ | ❌ |
| **AQLM 1×16** | ~4-bit | uint16 indices | (d_out, K//16) uint16 | (d_out,) fp16 | — | (1, 65536, 16) fp16 = 2MiB | ❌ | ❌ |
| **QTIP 1MAD/3INST** | 2-bit | k-bit packed | (K*k//8, N) uint8 | () fp32 | — | 0（纯计算） | ✅ | ❌ |
| **QTIP HYB** | 2-bit | k-bit packed | (K*k//8, N) uint8 | () fp32 | — | (512, 2) fp16 = 2KiB | ✅ | ❌ || **GGUF Q3_K** | 2bit+1high bit 标量 | hmask+qs+scales+d | fp16 d + int8 scales | — | — | ✅ | ❌ |
| **GGUF IQ3_XXS** | 8-bit index + 7-bit sign | qs(uint8) + scales_signs | fp16 d | 256×4D fp32 = 1KiB | 1KiB D4 codebook | ✅ | ❌ |
| **GGUF IQ3_S** | 9-bit index + 8-bit sign | qs+qh+signs+scales | fp16 d | 512×4D fp32 = 2KiB | 2KiB D4 codebook | ✅ | ❌ |
| **GGUF TQ1_0** | base-3 ternary (5/byte) | qs+qh uint8 | fp16 d | — | — | ✅ | ❌ |
| **GGUF TQ2_0** | 2-bit ternary (4/byte) | qs uint8 | fp16 d | — | — | ✅ | ❌ |
**注：** gs = group_size（通常 64 或 128）；K = in_features；N = out_features；r = LoRA rank

---

## 七、解压流程伪代码

### 7.1 通用标量量化解压（W3/W4，适用于 GPTQ/SE/ParetoQ/LR-QAT）

```python
def scalar_dequant_w4(qweight, scales, qzeros, group_size=128):
    """
    qweight: (K//8, N) int32
    scales:  (K//group_size, N) fp16
    qzeros:  (K//group_size, N//8) int32
    返回：(K, N) fp16
    """
    K = qweight.shape[0] * 8
    N = qweight.shape[1]
    W = torch.empty(K, N, dtype=torch.float16)

    for k in range(K):
        for n in range(N):
            # 解包 INT4
            word = qweight[k // 8, n]
            bit_offset = (k % 8) * 4
            w_int = (word >> bit_offset) & 0xF          # 4 bits

            # 解包 zero_point
            zp_word = qzeros[k // group_size, n // 8]
            zp_offset = (n % 8) * 4
            zp = (zp_word >> zp_offset) & 0xF

            # dequant
            scale = scales[k // group_size, n]
            W[k, n] = (w_int - zp) * scale
    return W

# CUDA kernel 等价（向量化版本）：
# 每个 thread 处理 8 个连续 k 的权重（1 个 int32 word）
# 使用 __shfl_down_sync 在 warp 内规约 dot product
```

### 7.2 QuIP# E8P 解压

```python
S_BASE = load_constant_codebook()  # (256, 8) fp16, 4KiB

def decode_e8p_codeword(cw: int) -> np.ndarray:
    """cw: uint16 codeword → 8 FP16 weights"""
    base_idx  = (cw >> 8) & 0xFF         # 8 bits → index into S_BASE
    sign_bits = (cw >> 1) & 0x7F         # 7 bits → 7 sign flips
    shift_flag = cw & 0x1                # 1 bit  → ±0.25 shift

    vec = S_BASE[base_idx].copy()        # (8,) fp16

    # Apply 7 sign flips (8th sign is parity-derived)
    parity = 1
    for i in range(7):
        if (sign_bits >> i) & 1:
            vec[i] *= -1
            parity *= -1
    vec[7] *= parity  # E8 parity constraint

    if shift_flag:
        vec += 0.25

    return vec

def quip_dequant_matvec(indices, S_U, S_V, x):
    """
    indices: (K//8, N//8) uint16
    S_U, S_V: sign vectors (packed bits)
    x: (K,) activation
    """
    # Step 1: apply V-transform to activation
    x_hat = x * unpack_signs(S_V)   # element-wise ±1

    # Step 2: decode weights and compute dot product
    y_hat = np.zeros(N // 8)
    for k8 in range(K // 8):
        for n8 in range(N // 8):
            cw  = indices[k8, n8]
            vec = decode_e8p_codeword(cw)         # 8 FP16 weights
            y_hat[n8] += np.dot(vec, x_hat[k8*8:(k8+1)*8])

    # Step 3: apply U^T-transform to output
    y = y_hat * unpack_signs(S_U)
    return y
```

### 7.3 AQLM 2×8 解压

```python
def aqlm_2x8_dequant_matvec(qcodes, codebooks, scales, x):
    """
    qcodes:     (d_out, K//8, 2) uint8
    codebooks:  (2, 256, 8) fp16   ← 8KiB total, L1-resident
    scales:     (d_out,) fp16
    x:          (K,) activation
    """
    d_out = qcodes.shape[0]
    K     = qcodes.shape[1] * 8
    y = np.zeros(d_out, dtype=np.float16)

    for n in range(d_out):
        acc = 0.0
        for k8 in range(K // 8):
            b1 = qcodes[n, k8, 0]      # uint8 index into codebook 0
            b2 = qcodes[n, k8, 1]      # uint8 index into codebook 1

            vec = codebooks[0, b1] + codebooks[1, b2]  # 8-wide fp16 add
            acc += np.dot(vec, x[k8*8:(k8+1)*8])

        y[n] = acc * scales[n]
    return y
```

### 7.4 QTIP 3INST 解压

```python
# Constants
A3 = 89226354
B3 = 64248484
MAGIC_FP16 = 0x3B60  # ≈ 0.922 in fp16 bit pattern

def qtip_3inst_decode_weight(trellis_bits: int) -> float:
    """
    trellis_bits: uint16（连续 L=16 bit 的格码状态）
    返回：1 个 FP16 权重
    """
    # Step 1: LCG update (1 MAD instruction)
    x = (A3 * trellis_bits + B3) & 0xFFFFFFFF

    # Step 2: FP16 bit manipulation (1 lop3 instruction in CUDA)
    lo16 = x & 0xFFFF
    hi16 = (x >> 16) & 0xFFFF
    m1_bits = lo16 ^ MAGIC_FP16   # XOR with magic constant
    m2_bits = hi16 ^ MAGIC_FP16
    m1 = bits_to_fp16(m1_bits)
    m2 = bits_to_fp16(m2_bits)

    # Step 3: FP16 add (1 FADD instruction)
    w = fp16_add(m1, m2)
    return w

def qtip_dequant_matvec(packed, sign_U, sign_V, x, scale):
    """
    packed:  (K * 2 // 8, N) uint8  ← k=2 bits/weight, bit-packed
    sign_U:  (N // 8,) uint8        ← S_U packed bits
    sign_V:  (K // 8,) uint8        ← S_V packed bits
    """
    # Step 1: Apply V-transform to activation
    x_hat = x * unpack_signs(sign_V)

    # Step 2: Decode weights + GEMV
    y_hat = np.zeros(N)
    for n in range(N):
        acc = 0.0
        for k16 in range(K // 16):
            # Load 16 packed bits (2 bytes for k=2: 8 weights use 16 bits)
            trellis = load_u16(packed, k16, n)  # 连续 16 bits

            # Decode 1 weight (V=1)
            w = qtip_3inst_decode_weight(trellis)
            w *= scale
            acc += w * x_hat[k16]  # 注意：此处简化，实际 V 个权重对应 V 个 x 值
        y_hat[n] = acc

    # Step 3: Apply U-transform to output
    y = y_hat * unpack_signs(sign_U)
    return y
```

### 7.5 LoftQ/PiSSA 推理（带 LoRA 分支）

```python
def loftq_inference(x, W_q, scales_q, qzeros, A, B, lora_alpha, lora_r):
    """
    x:        (batch, K) fp16
    W_q:      (K//8, N) int32 (INT4 packed)
    A:        (K, r) fp16
    B:        (r, N) fp16
    """
    # Branch 1: 量化主路径（on-the-fly dequant + GEMM）
    W_fp = scalar_dequant_w4(W_q, scales_q, qzeros)   # (K, N) fp16（kernel 内不物化）
    main_out = x @ W_fp                                 # (batch, N)

    # Branch 2: LoRA 路径（FP16 GEMM × 2）
    lora_mid = x @ B              # (batch, r)，通常 r≤64
    lora_out = lora_mid @ A       # (batch, K)... 注意维度顺序按实现而定
    # 等价：lora_out = (x @ B) @ A = x @ (B @ A)

    # Merge
    scale = lora_alpha / lora_r
    y = main_out + scale * lora_out
    return y  # (batch, N)
```

---

## 八、OpenVINO 实现建议

### 8.1 现有 GPU Plugin 扩展点

OpenVINO GPU Plugin（`src/plugins/intel_gpu/`）中，3-bit 解压 kernel 的最佳插入点：

```
src/plugins/intel_gpu/src/kernel_selector/
├── core/
│   ├── cl_kernels/
│   │   ├── gemm_mmad_int8.cl         ← W8A8 参考
│   │   ├── fully_connected_gpu_bf16.cl ← FP16 FC 参考
│   │   └── [NEW] fc_3bit_dequant.cl  ← 新增 3-bit dequant kernel
│   └── kernels/
│       └── fully_connected/
│           ├── fully_connected_kernel_base.cpp
│           └── [NEW] fully_connected_kernel_3bit.cpp
└── transformations/
    └── [NEW] convert_weights_3bit.cpp ← 量化权重转换 pass
```

### 8.2 各算法 kernel 实现优先级

```
优先级 1（最简单，重用现有 scalar dequant）：
  ■ GPTQ W3/W4    → 扩展现有 INT4 kernel，改 shift 参数
  ■ SE+AWQ W3/W4  → 在 scale 融合时乘 per-channel scale（1 mul overhead）
  ■ LR-QAT/NNCF  → 同 GPTQ W3/W4
  ■ ParetoQ       → 同 GPTQ W3/W4
  ■ BitNet b1.58  → 新 kernel，int8 unpack，但逻辑简单

优先级 2（需要 L1 codebook 管理）：
  ■ AQLM 2×8     → 需保证 8KiB codebook 在 L1 (OCL: __constant 或 __local)
  ■ QuIP#        → 需 1KiB S_BASE 常量内存 + RHT sign 向量 pre-processing

优先级 3（需要 ALU codepath 特化）：
  ■ QTIP 3INST   → 纯 ALU，需要内联 LCG + FP16 bit manipulation
                   OpenCL 中用 as_half2() 做 bit reinterpret

优先级 4（需要 LoRA 分支融合）：
  ■ LoftQ/PiSSA  → 需要 INT4 GEMM + FP16 LoRA GEMM 两路合并 kernel
```

### 8.3 OpenCL Kernel 伪代码框架（GPTQ W3/W4 基础）

```c
// fully_connected_kernel_3bit.cl

__kernel void fc_dequant_gemv_w4(
    __global const int*   qweight,     // (K/8, N) int32
    __global const half*  scales,      // (K/gs, N) half
    __global const int*   qzeros,      // (K/gs, N/8) int32
    __global const half*  x,           // (K,) half
    __global       half*  y,           // (N,) half
    const int K, const int N, const int group_size)
{
    const int n = get_global_id(0);    // output channel
    if (n >= N) return;

    float acc = 0.0f;

    for (int k8 = 0; k8 < K / 8; k8++) {
        int word  = qweight[k8 * N + n];          // coalesced load
        int k_base = k8 * 8;

        int g = k_base / group_size;
        float scale = (float)scales[g * N + n];

        int zp_word = qzeros[g * (N/8) + n/8];
        int zp = (zp_word >> ((n % 8) * 4)) & 0xF;

        // 展开 8 个 INT4
        for (int i = 0; i < 8; i++) {
            int w_int = (word >> (i * 4)) & 0xF;
            float w_fp = ((float)(w_int - zp)) * scale;
            acc += w_fp * (float)x[k_base + i];
        }
    }

    y[n] = (half)acc;
}
```

### 8.4 AQLM 2×8 OpenCL 框架

```c
// fc_aqlm_2x8.cl — 关键：确保 codebook 进入 __constant memory（L1）

__constant half cb0[256][8];   // codebook 0: 4KiB, 编译期烧入
__constant half cb1[256][8];   // codebook 1: 4KiB

__kernel void fc_aqlm_2x8_gemv(
    __global const uchar2* qcodes,  // (d_out, K/8) uchar2
    __global const half*   scales,  // (d_out,) half
    __global const half*   x,       // (K,) half
    __global       half*   y,       // (d_out,) half
    const int K, const int d_out)
{
    int n = get_global_id(0);
    if (n >= d_out) return;

    float acc = 0.0f;
    for (int k8 = 0; k8 < K / 8; k8++) {
        uchar2 codes = qcodes[n * (K/8) + k8];
        uchar b1 = codes.x, b2 = codes.y;

        for (int i = 0; i < 8; i++) {
            // codebook lookup（L1 __constant → 快）
            float w = (float)cb0[b1][i] + (float)cb1[b2][i];
            acc += w * (float)x[k8 * 8 + i];
        }
    }

    y[n] = (half)(acc * (float)scales[n]);
}
```

### 8.5 QTIP 3INST OpenCL 框架

```c
// fc_qtip_3inst.cl — 关键：as_half2() 做 bit reinterpretation

#define QTIP_A 89226354u
#define QTIP_B 64248484u
#define QTIP_MAGIC 0x3B60   // ≈ 0.922 in fp16

inline float qtip_3inst(uint trellis) {
    // Step 1: LCG
    uint x = QTIP_A * trellis + QTIP_B;

    // Step 2: FP16 bit manipulation via as_half2
    ushort lo = (ushort)(x & 0xFFFF) ^ QTIP_MAGIC;
    ushort hi = (ushort)(x >> 16)    ^ QTIP_MAGIC;
    half m1 = as_half(lo);
    half m2 = as_half(hi);

    // Step 3: FP16 add
    return (float)(m1 + m2);
}

__kernel void fc_qtip_3inst_gemv(
    __global const uchar* packed,   // (K*2/8, N) uint8, k=2 bits/weight
    __global const uchar* sign_U,   // (N/8,) uint8
    __global const uchar* sign_V,   // (K/8,) uint8
    __global const half*  x,        // (K,) half
    __global       half*  y,        // (N,) half
    const float scale, const int K, const int N)
{
    int n = get_global_id(0);
    if (n >= N) return;

    float acc = 0.0f;
    for (int k16 = 0; k16 < K / 16; k16++) {
        // Load 16 packed bits (k=2: 16 weights use 32 bits = 4 bytes)
        // 简化：取 16-bit trellis 状态 (2 bytes per V=1 weight)
        uint trellis = ((uint)packed[(k16 * 2) * N/4 + n/4]) |  // 简化寻址
                       ((uint)packed[(k16 * 2 + 1) * N/4 + n/4] << 8);

        float w = qtip_3inst(trellis) * scale;

        // Apply V-transform sign
        int sign_v_bit = (sign_V[k16 / 8] >> (k16 % 8)) & 1;
        if (sign_v_bit) w = -w;

        acc += w * (float)x[k16];
    }

    // Apply U-transform sign
    int sign_u_bit = (sign_U[n / 8] >> (n % 8)) & 1;
    y[n] = (half)(sign_u_bit ? -acc : acc);
}
```

### 8.6 性能预期汇总

```
算法               推理吞吐（估算，相对 FP16 baseline）  实现难度
───────────────────────────────────────────────────────────────
GPTQ W4 (INT4)    2–3× 加速                              ★☆☆☆
GPTQ W3 (INT3)    1.5–2× 加速（packing 复杂度略高）       ★★☆☆
BitNet b1.58      2–4× 加速（W1.58A8 特殊 kernel）        ★★☆☆
AQLM 2×8          1.3× 加速（需保证 L1 cache）            ★★★☆
QuIP# E8P         ~2× 加速（RHT + codebook lookup）       ★★★★
QTIP 3INST        ~2× 加速（纯 ALU，需 FP16 bit hack）    ★★★☆
LoftQ/PiSSA       ≈ GPTQ W4（LoRA overhead < 1%）         ★★☆☆
AQLM 1×16         慢于 FP16（不建议使用）                  ——
GGUF Q3_K        1.5–2× 加速（与 GPTQ W3 几乎相同）       ★★☆☆
GGUF IQ3_S       ~1.4× 加速（2KiB D4 LUT + sign flip）    ★★★☆
GGUF IQ3_XXS     ~1.3× 加速（1KiB D4 LUT + sign flip）    ★★★☆
GGUF TQ1_0/2_0   2–3× 加速（base-3 unpack，BitNet 专用）   ★★☆☆
```

---

## 九、GGUF 3-bit 格式解压细节

> 来源：llama.cpp `ggml-quants.c`；Unsloth Qwen3.6-35B-A3B-GGUF；SUMMARY.md §十

GGUF 的 3-bit 格式在 CPU/GPU 解压方面与 GPTQ/QuIP# 有所不同：全部以 **super-block（QK_K=256 个权重）** 为量化单元，内部再分 sub-block 管理 scale。

---

### 9.1 Q3_K — k-quants 标量量化

**等效范式：** 范式 A（标量），与 GPTQ W3 原理相同但打包格式不同。

#### 内存布局（`block_q3_K`，256 个权重/block）

```c
typedef struct {
    uint8_t  hmask[32];    // 高 1 bit（bit-2），每权重 1 bit，bitpack
    uint8_t  qs[64];       // 低 2 bits，4 个权重/byte（2 bits each）
    int8_t   scales[12];   // 16 个 sub-block 的 6-bit scale，位交织打包
    ggml_fp16_t d;         // super-block scale（fp16）
} block_q3_K;
// 总计：2 + 12 + 64 + 32 = 110 bytes / 256 weights = 3.4375 bits/weight
```

#### 解压流程

```
HBM (block_q3_K)
      │
      ▼ 加载 110 bytes（256 权重的全部数据）
寄存器
      │
      ▼ 每个权重的 3-bit 值重组：
        low2  = (qs[k/4]  >> ((k%4)*2)) & 0x3        // 低 2 bits
        high1 = (hmask[k/8] >> (k%8))   & 0x1        // 高 1 bit
        val3  = low2 | (high1 << 2)                  // 3-bit 无符号 [0..7]
        int_w = val3 - 4                              // → 有符号 [-4..+3]
      │
      ▼ 应用 sub-block scale：
        sub_scale = 解码 scales[12]（6-bit 位交织，16个sub-block）
        dl = d * (sub_scale - 32)                     // sub-block 实际 scale
        w_fp = dl * int_w                             // float 权重
```

#### Q3_K vs GPTQ W3 解压对比

```
特征                 GPTQ W3              GGUF Q3_K
─────────────────────────────────────────────────────────────────
bits/weight          3.0                  3.4375
group_size           128 (configurable)   16 (fixed sub-block)
scale bits           16 bit (fp16)        6 bit（量化的 scale）
zero_point           有（packed INT3）     无（对称，offset=-4）
bits 存储             perfect 3-bit pack   2+1 分离（qs+hmask）
跨 int32 问题         有（每 10 or 32/3）   无（low/high 分开存）
解压指令数           5–6                  5–6（相当）
```

**关键差异：** Q3_K 将低 2-bit 和高 1-bit 分开存储（`qs` 和 `hmask`），避免了 GPTQ W3 完美打包时的跨 int32 边界问题，但需要两次内存读取合并。scale 使用 6-bit 量化（而非 fp16），节省了 scale 存储空间（12 bytes vs 通常的 32 bytes）。

#### sub-block scale 解码（6-bit 位交织）

```c
// scales[12] 中存储 16 个 sub-block 的 6-bit scale
// 位布局（来自 ggml-quants.c）：
// j < 8:  scales[j] = lower 6 bits
// j >= 8: scales[j] = upper 4 bits | lower 2 bits from another field
void get_scale_min_k4(int j, const uint8_t * q, uint8_t * d) {
    if (j < 4) {
        *d = q[j] & 63;
    } else {
        *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
    }
}
// 最终：dl = d_all * (decoded_scale - 32)  ← 中心化到 [-32, +31]
```

#### Q3_K_S / Q3_K_M / Q3_K_XL 解压差异

```
Q3_K_S:  所有层均为 Q3_K，解压逻辑统一
Q3_K_M:  关键层（attention QKV、FFN gate）升级为 Q4_K 格式
          → Q4_K 解压：每 32 权重存于 block_q4_K，4-bit + min + scale
          → 同一 kernel 需支持 Q3_K 和 Q4_K 两种路径
Q3_K_XL: 同 Q3_K_M 格式，但量化时使用 imatrix 加权优化 rounding
          → 推理时解压完全相同（imatrix 只影响量化阶段，不影响推理）
```

---

### 9.2 IQ3_XXS — 3.0625 bpw D4 lattice 向量量化

**等效范式：** 范式 B（向量量化），codebook = 256-entry 4D D4 lattice 子集（~1KiB）。

#### 内存布局（`block_iq3_xxs`，256 个权重/block）

```c
typedef struct {
    ggml_fp16_t d;              // 2 bytes: super-block scale
    uint8_t  qs[3*QK_K/8];     // 96 bytes: 8-bit codebook index（每 4 weights 一个 uint8）
    // 内嵌在 qs 末尾的 scales_and_signs（每 32 weights 一个 uint32）：
    // bits[31:28] = 4-bit sub-block scale (value 0–15)
    // bits[27:21] = 7-bit sign mask (group 0，4 weights → 但只用 7 bits)
    // bits[20:14] = 7-bit sign mask (group 1)
    // bits[13: 7] = 7-bit sign mask (group 2)
    // bits[ 6: 0] = 7-bit sign mask (group 3)
} block_iq3_xxs;
// 总计：2 + 96 + (4×8) = 130 bytes ≈ 3.0625 bpw
```

#### D4 Lattice Codebook 结构

```
kgrid_256[256]（编译期常量，~1KiB）：
  每个 entry = 4D 向量，每维取奇数值 {1,3,5,7,9,11,13,15}（3 bits/维）
  → 256 = 2^8 个 entry（8-bit index 可直接寻址）
  → 等效于 4D 空间中均匀分布的点（D4 lattice 子集）

实际权重范围（经 scale 后）：
  理论：±15 × db（db = d × (0.5 + sub_scale) × 0.25）
  每组 4 个权重取奇数值 → 量化误差约为 1×db（比标量 INT3 大）

7-bit sign mask 扩展码本：
  每个 4D vector 可独立翻转 7 个分量符号（第 8 个由奇偶性约束）
  等效码本大小：256 × 128 = 32768 个不同的 4D 向量
```

#### 解压流程（per 32-weight block）

```
HBM: 每 32 weights 读取 8 bytes（qs[8]） + 4 bytes（scales_and_signs）

Step 1：解码 sub-block scale 和 sign masks
  aux32 = scales_and_signs[ib32]                  // 1 uint32
  db    = d * (0.5f + (aux32 >> 28)) * 0.5f      // sub-block scale
  sign0 = (aux32 >>  0) & 0x7F                    // 7-bit sign for group 0
  sign1 = (aux32 >>  7) & 0x7F
  sign2 = (aux32 >> 14) & 0x7F
  sign3 = (aux32 >> 21) & 0x7F

Step 2：每 8 个权重（2 个 4D groups）解压
  for l in 0..3:                                  // 4 个 4D groups per 32-weight block
      idx1   = qs[2*l + 0]                        // 8-bit index
      idx2   = qs[2*l + 1]
      grid1  = kgrid_256[idx1]                    // L1 LUT 查表，4 个 int8（奇数值）
      grid2  = kgrid_256[idx2]
      signs  = sign_mask[l]                       // 7-bit

      for j in 0..3:                              // 4D vector 的 4 个分量
          y[j+0] = db * grid1[j] * (signs & kmask[j+0] ? -1 : +1)
          y[j+4] = db * grid2[j] * (signs & kmask[j+4] ? -1 : +1)
```

#### L1 Cache 分析

```
codebook kgrid_256[256]：
  256 × 4 bytes（4D int8 向量）= 1KiB 原始大小
  × 32（GPU warp bank 冲突展开）≈ 32KiB
  H100 L1 = 256KB → ✅ 充足（与 AQLM 2×8 的 8KiB 相当）

每 8 weights 的指令数：
  2× uint8 load（qs）+ 2× L1 lookup（grid）+ 7-bit sign mask + 8× mul = ~8 instructions
  平均 ~1 instruction/weight（比标量 Q3_K 略低，因为向量化程度高）
```

#### 与 QuIP# E8P 的对比

```
特征               QuIP# E8P         IQ3_XXS
──────────────────────────────────────────────────────────
codebook 维数       8D               4D
codebook 大小       256×8 fp16=4KiB  256×4 int8=1KiB
码字 bits          16 bits/8D       8 bits/4D (+ 7-bit sign)
有效码本大小         2^16 逻辑（解码） 256×128=32768
bpw（有效）         ~2 bit           3.0625 bit
Hadamard 变换      需要（RHT）        不需要
解压指令数          <5/weight         ~1–2/weight（4D 向量化）
```

---

### 9.3 IQ3_S — 3.3125 bpw（512-entry D4，更高精度）

**相比 IQ3_XXS 的改进：** 9-bit index（vs 8-bit）+ 8-bit sign mask（vs 7-bit）。

#### 内存布局（`block_iq3_s`，256 个权重/block）

```c
typedef struct {
    ggml_fp16_t d;            // 2  bytes: super-block scale
    uint8_t  qs[QK_K/4];     // 64 bytes: 低 8 bits of 9-bit codebook index（每 4 weights）
    uint8_t  qh[QK_K/32];    // 8  bytes: 高 1 bit of 9-bit index（每 8 entries 共享 1 byte）
    uint8_t  signs[QK_K/8];  // 32 bytes: 8-bit sign mask（每 8 weights = 2 个 4D groups）
    uint8_t  scales[QK_K/32];// 8  bytes: 4-bit sub-block scale pairs（每 byte 存 2 个）
} block_iq3_s;
// 总计：2 + 64 + 8 + 32 + 8 = 114 bytes / 256 weights = 3.5625 bpw（官方标注 3.3125 为纯权重 bpw）
```

#### 512-entry Codebook

```
iq3s_grid[512]（~2KiB）：
  512 = 2^9 个 entry，9-bit index 寻址
  每个 entry = 4D int8 向量（奇数值 {1,3,...,15}）
  相比 256-entry：码本粒度更细，4D 空间覆盖更密

9-bit index 的存储：
  低 8 bits → qs[k/4]（每个 byte 存 1 个 index，4 weights/index）
  高 1 bit  → qh[k/32] 的对应 bit（每 byte 存 8 个高位）
  重组：idx = qs[2*l+0] | ((qh[0] << (8 - 2*l)) & 256)
```

#### 解压流程（per 32-weight block）

```
HBM: 读取 8+1+4+1 = 14 bytes per 32 weights（更密集）

Step 1：解码 sub-block scale
  db = d * (1 + 2*(scales[ib32/2] & 0xf))    // 奇数值 scale {1,3,...,31}

Step 2：每 8 个权重（2 个 4D groups）解压
  for l in 0..3:                              // 4 个 4D groups per 32-weight block
      // 重组 9-bit index
      idx1 = qs[2*l+0] | ((qh[0] << (8-2*l)) & 256)   // 9 bits
      idx2 = qs[2*l+1] | ((qh[0] << (7-2*l)) & 256)

      grid1 = iq3s_grid[idx1]    // 512-entry L1 lookup，4D int8
      grid2 = iq3s_grid[idx2]
      sign_byte = signs[l]       // 8-bit（vs IQ3_XXS 的 7-bit）

      for j in 0..3:
          y[j+0] = db * grid1[j] * (sign_byte & kmask[j+0] ? -1 : +1)
          y[j+4] = db * grid2[j] * (sign_byte & kmask[j+4] ? -1 : +1)
```

#### IQ3_S vs IQ3_XXS 解压性能对比

```
特征              IQ3_XXS          IQ3_S
───────────────────────────────────────────────────────
codebook 大小     1KiB (256)       2KiB (512)
bpw               3.0625           3.3125（更大文件）
index bits        8                9（需要 qh 额外字段）
sign bits         7/4D-group       8/4D-group（更多符号自由度）
sub-block scale   4-bit (0–15)     4-bit → 奇数 {1,3,...,31}
L1 cache          ✅               ✅（2KiB，仍可 fit）
ppl（Llama-3-8B） ~7.2             ~6.8（精度略好）
解压指令数        相近（~8/32w）    相近（+qh 重组 ~1-2 instr 额外）
```

---

### 9.4 TQ1_0 / TQ2_0 — GGUF 三值量化（BitNet 专用）

**适用模型：** BitNet b1.58 及 TriLM 系列（Qwen3.6 GGUF 中**不包含**此格式）。

#### TQ1_0（1.6875 bpw，base-3 packed）

```c
typedef struct {
    ggml_fp16_t d;          // super-block scale
    uint8_t  qs[QK_K/5+ggml_pad(QK_K/5, 32)]; // base-3 packed，5 ternary/byte
    uint8_t  qh[QK_K/4];   // 4 ternary/byte（尾部溢出）
} block_tq1_0;
```

**编码方式（base-3 打包）：**
```
5 个三值权重 {-1,0,+1} → 映射为 {0,1,2} → 相加构成 base-3 数
  encoded = w[0] + w[1]*3 + w[2]*9 + w[3]*27 + w[4]*81
  存入 1 byte（0–242，3^5=243 ≤ 256）

3^5=243 不整除 256 → 实际每 byte 浪费 (256-243)/256 ≈ 5%
```

**解压算法（从 ggml-quants.c 直接推导）：**
```c
// 解压 1 个 byte → 5 个 ternary 权重
// pow3 = {1, 3, 9, 27, 81, 243}
uint8_t q = qs[j] * pow3[n];         // 乘以 3^n
int16_t xi = ((uint16_t)q * 3) >> 8; // q*3/256 ≈ q/85，取商 0/1/2
float w = (float)(xi - 1) * d;       // {0,1,2} → {-1,0,+1} × d
```

**解压效率：**
```
每权重约 3 条指令（乘法 + 右移 + 减法）
无 LUT，纯 ALU，register 压力低
关键 trick：uint8 乘法 + 右移 8 位 ≈ base-3 除法（利用整数近似）
```

#### TQ2_0（2.0 bpw，2-bit ternary）

```c
typedef struct {
    ggml_fp16_t d;          // super-block scale
    uint8_t  qs[QK_K/4];   // 4 ternary/byte，2 bits each
} block_tq2_0;
```

**编码方式：** 直接 2-bit 打包，{-1→0, 0→1, +1→2}

**解压：**
```c
// 解压 4 个 ternary 权重（1 byte）
for l in 0..3:
    q   = (qs[j] >> (l*2)) & 0x3    // 2-bit 提取
    w   = (float)(q - 1) * d        // {0,1,2} → {-1,0,+1}
```

**相比 TQ1_0：** 更简单（直接 2-bit 解包），但 bpw 略高（2.0 vs 1.69）；实际 llama.cpp 中 TQ2_0 用于对齐或备用场景。

---

### 9.5 GGUF 3-bit 格式 OpenCL 解压框架

#### Q3_K OpenCL Kernel（与 GPTQ W3 类似，主要差异在 hmask）

```c
// gguf_q3k_dequant_gemv.cl
__kernel void q3k_dequant_gemv(
    __global const uchar* hmask,    // (K/8, N/block_n) — 高 bit
    __global const uchar* qs,       // (K/4, N/block_n) — 低 2 bits
    __global const char*  scales,   // (12, N/block_n) — 6-bit sub-block scales
    __global const half*  d,        // (N/block_n,) — super-block fp16 scale
    __global const half*  x,        // (K,) activation
    __global       half*  y,        // (N,) output
    const int K, const int N)
{
    int n = get_global_id(0);
    if (n >= N) return;

    float acc = 0.f;
    const float d_val = (float)d[n / 256];
    uint8_t m = 1;  // hmask bit selector

    for (int k = 0, is = 0; k < K; k += 16, is++) {
        // 解码 6-bit sub-block scale（简化）
        int sc_raw = decode_q3k_scale(scales, n, is);       // -32..+31
        float dl   = d_val * (float)(sc_raw - 32);

        for (int l = 0; l < 16; l++) {
            int kidx = k + l;
            // 低 2 bits
            int low2  = (qs[kidx/4 + n*(K/4)] >> ((kidx%4)*2)) & 0x3;
            // 高 1 bit（来自 hmask）
            int high1 = (hmask[kidx/8 + n*(K/8)] >> (kidx%8)) & 0x1;
            int val   = low2 | (high1 << 2);         // 3-bit 值 [0..7]
            float w   = dl * (float)(val - 4);       // [-4..+3]
            acc += w * (float)x[kidx];
        }
    }
    y[n] = (half)acc;
}
```

#### IQ3_S OpenCL Kernel（D4 lattice，__constant codebook）

```c
// gguf_iq3s_dequant_gemv.cl
// 512-entry codebook，编译期常量（2KiB）
__constant uchar iq3s_grid_flat[512*4] = { /* ... 预展开的 int8 向量 ... */ };

inline float4 lookup_iq3s(uint idx) {
    // 每个 entry = 4 个 int8（奇数值）
    const __constant uchar* v = iq3s_grid_flat + idx * 4;
    return (float4)((float)v[0], (float)v[1], (float)v[2], (float)v[3]);
}

__kernel void iq3s_dequant_gemv(
    __global const uchar* qs,      // 低 8 bit index
    __global const uchar* qh,      // 高 1 bit
    __global const uchar* signs,   // 8-bit sign mask
    __global const uchar* scales,  // 4-bit sub-block scale pairs
    __global const half*  d,       // super-block scale
    __global const half*  x,
    __global       half*  y,
    const int K, const int N)
{
    int n = get_global_id(0);
    if (n >= N) return;

    float acc = 0.f;
    const float d_val = (float)d[n / 256];

    for (int ib32 = 0; ib32 < K/32; ib32++) {
        // sub-block scale（奇数值 1,3,5,...,31）
        uchar sc_byte = scales[(ib32/2) + n*(K/64)];
        float db = (ib32 % 2 == 0)
            ? d_val * (1 + 2*(sc_byte & 0xf))
            : d_val * (1 + 2*(sc_byte >> 4));

        uchar qh_byte0 = qh[(ib32*2)   + n*(K/128)];
        uchar qh_byte1 = qh[(ib32*2+1) + n*(K/128)];

        for (int l = 0; l < 4; l++) {
            // 重组 9-bit index
            uint idx1 = qs[(ib32*8+2*l+0)] | (((uchar)(qh_byte0 << (8-2*l))) & 256);
            uint idx2 = qs[(ib32*8+2*l+1)] | (((uchar)(qh_byte0 << (7-2*l))) & 256);

            float4 g1 = lookup_iq3s(idx1);
            float4 g2 = lookup_iq3s(idx2);
            uchar  sg = signs[ib32*4 + l];

            // 8-bit sign mask 应用
            for (int j = 0; j < 4; j++) {
                float w1 = db * g1[j] * ((sg & (1<<(j+0))) ? -1.f : 1.f);
                float w2 = db * g2[j] * ((sg & (1<<(j+4))) ? -1.f : 1.f);
                acc += w1 * (float)x[ib32*32 + l*8 + j + 0];
                acc += w2 * (float)x[ib32*32 + l*8 + j + 4];
            }
        }
    }
    y[n] = (half)acc;
}
```

---

### 9.6 GGUF 3-bit 格式与其他算法的解压对比

| 格式 | bpw | 解压指令数/weight | codebook | HBM 读取（per weight） | L1-safe |
|---|---|---|---|---|---|
| GPTQ W3 | 3.0 | 5–6 | — | 3/8 byte | ✅ |
| **Q3_K** | 3.4375 | 5–6 | — | ~3.4/8 byte | ✅ |
| **IQ3_XXS** | 3.0625 | ~2（向量化） | 256×4D（1KiB） | ~3.1/8 byte | ✅ |
| **IQ3_S** | 3.3125 | ~2–3 | 512×4D（2KiB） | ~3.6/8 byte | ✅ |
| QuIP# E8P | ~2 | <5 | 256×8D（4KiB） | ~2/8 byte | ✅ |
| AQLM 2×8 | ~2 | ~4 | 2×256×8D（8KiB） | ~2/8 byte | ✅ |
| **TQ1_0** | 1.69 | ~3 | — | ~1.7/8 byte | ✅ |
| **TQ2_0** | 2.0 | ~2 | — | 2/8 byte | ✅ |

**结论：**
- Q3_K 是三种 GGUF 标量格式中实现最简单的，与 GPTQ W3 几乎等价，适合优先支持
- IQ3_S/IQ3_XXS 引入了 D4 lattice codebook（2KiB/1KiB），需要 GPU `__constant` 内存配置，但精度优于 Q3_K_S 约 0.3–0.5 ppl
- TQ1_0/TQ2_0 专为 BitNet 设计，base-3 解压逻辑独特，但当前主流模型暂不使用

---

## 参考

- BitNet b1.58 paper: arXiv:2504.12285 (Ma et al., 2025)
- QuIP# paper: arXiv:2402.04396 (Tseng et al., 2024)  
- AQLM paper: arXiv:2401.06118 (Egiazarian et al., 2024)
- QTIP paper: arXiv:2406.11235 (Tseng et al., 2024)
- SignRoundV2: SUMMARY.md §八
- LoftQ/PiSSA/IR-QLoRA: SUMMARY.md §九
- GGUF Q3_K / IQ3_XXS / IQ3_S / TQ1_0 / TQ2_0: SUMMARY.md §十
- llama.cpp `ggml-quants.c`: github.com/ggml-org/llama.cpp
- Unsloth Qwen3.6-35B-A3B-GGUF: huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF
- OpenVINO GPU kernel references: `src/plugins/intel_gpu/src/kernel_selector/`
