# diffusion_gemma GPU Kernel 性能分析

> 范围：基于一次完整的 GPU kernel 级 trace（`clintercept_trace.json`，CLIntercept 采集，
> Chrome Trace 格式，14 MB / 108,086 行）分析 `modeling_diffusion_gemma.exe` 在 GPU 上每个
> OpenCL kernel 的**执行次数、单次/累计耗时**，定位计算瓶颈与优化方向。
> 设备：Intel(R) Arc(TM) B390 GPU（多 OOQ/IOQ 队列）。
> 模型/命令：`modeling_diffusion_gemma.exe --model diffusiongemma-26B-A4B-it --device GPU
> --prompt "Why is the sky blue?" --output-tokens 256`（INT4-asym 在途量化 MoE，融合路径）。
> 采集口径：trace 跨度含模型编译 + encoder + warmup + 48 步去噪生成。
> 日期：2026-06-22

---

## 0. TL;DR

- **GPU 利用率仅 4.5%**：trace 跨度 **435.9 s**，但 GPU 实际忙碌仅 **19.4 s**，
  **416.5 s 在空转** → **头号瓶颈是 host-bound（CPU 串行后处理阻塞 GPU），不是 kernel 本身**。
- **GPU 忙碌时间里 MoE 专家 GEMM 独占 58.5%**（`grouped_micro_gemm`，11.37 s / 4320 次），
  但实测算力仅 ~2.3 TFLOP/s ≈ B390 峰值的 **~5%** → 小 GEMM 填不满 GPU，访存受限。
- **48 步去噪是所有成本的乘数**：decoder = 48 × 整模型前向；1440 次 decoder MoE 调用
  = 48 步 × 30 层（+30 encoder = 1470，与 `moe_router` 计数 1470 完全吻合）。
- **小 kernel 海**：17,568 个 eltwise + 13,010 个 reorder/permute，单个都很小，
  合计占 GPU 忙碌 **21%**，其中 reorder 纯属相邻 kernel layout 不匹配的浪费。
- **优先级**：① 修 host-bound 空转（投入产出比最高，砍 ~400 s wall-clock）→
  ② 融合 MoE gate+up GEMM → ③ 修 NaN 后减少去噪步数 → ④ 图融合消除小 kernel/reorder。

---

## 1. 总览数据

| 指标 | 值 |
|---|---|
| GPU 时间轴跨度 | **435.9 s** |
| GPU 实际忙碌 | **19.4 s** |
| **GPU 利用率** | **4.5%** ⚠️ |
| 空闲 / host-bound 间隙 | **416.5 s** |
| 设备 kernel 总启动数 | 108,041 |
| 不同 kernel 种类 | 142 |

> 注：trace 跨度 435 s 比之前 verbose 日志的 "denoising time 182 s"
> （见 [diffusion_gemma_flow_memory_cn.md](diffusion_gemma_flow_memory_cn.md)）更大，
> 是因为 trace 还含模型编译 + encoder + warmup。但 **4.5% 利用率**这个结论与口径无关，
> host-bound 是确定的。

---

## 2. GPU 忙碌时间的功能分布

按功能区聚合（占 GPU 忙碌 19.4 s 的比例）：

| 占比 | 时间 | 功能区 | 启动次数 | 平均 |
|---|---|---|---|---|
| **58.5%** | 11.37 s | **MoE 专家 GEMM** `grouped_micro_gemm` | 4,320 | 2632 us |
| 14.8% | 2.87 s | eltwise / activation / reduce / broadcast | ~33k | 小 |
| 10.4% | 2.01 s | dense GEMM `gemm_kernel`（attn q/k/v/o + dense MLP） | 9,258 | 217 us |
| 6.2% | 1.20 s | reorder / permute（布局转换） | ~18k | 小 |
| 4.3% | 0.83 s | MoE glue（gather/scatter/swiglu/router） | ~6k | 小 |
| 2.5% | 0.49 s | RMSNorm `rms_gpu` | 8,965 | 54 us |
| 1.9% | 0.36 s | SDPA attention `sdpa_micro__prefill` | 1,470 | 246 us |
| 0.8% | 0.15 s | dynamic_quantize | 7,632 | 19 us |
| 0.6% | 0.11 s | other | — | — |
| 0.2% | 0.04 s | rope | — | — |

### Top kernel（按累计耗时）

| 累计 | 次数 | 平均 | kernel |
|---|---|---|---|
| 11369.77 ms | 4320 | 2631.9 us | `grouped_micro_gemm` |
| 2012.90 ms | 9258 | 217.4 us | `gemm_kernel` |
| 443.37 ms | 1440 | 307.9 us | `moe_scatter_reduction_opt` |
| 362.44 ms | 2880 | 125.9 us | `generic_eltwise_ref` |
| 313.30 ms | 2892 | 108.3 us | `generic_eltwise_ref` |
| 309.97 ms | 5880 | 52.7 us | `reduce_ref` |
| 279.20 ms | 1225 | 227.9 us | `sdpa_micro__prefill` |
| 188.68 ms | 1470 | 128.3 us | `rms_gpu_ref` |
| 136.54 ms | 1440 | 94.8 us | `moe_3gemm_swiglu_fuse_prefill_swiglu` |
| 86.39 ms | 1440 | 60.0 us | `moe_gather_ref_prefill` |

---

## 3. 瓶颈 1（最致命）：GPU 95.5% 时间空转 —— host-bound

**435.9 s 里只有 19.4 s 在算，416.5 s 在等 CPU。** 这比任何 kernel 优化都重要。

**根因在 sample 的去噪循环**（`openvino.genai/src/cpp/src/modeling/samples/modeling_diffusion_gemma.cpp`）：
每次 decoder 前向之后，host 端对 **256 个位置 × vocab(≈256k)** 做了 4 轮单线程串行循环：

- `extract_logits_f32`（256 × vocab 拷贝 + dtype 转换）
- `apply_temperature_schedule`（256 × vocab 乘法）
- `compute_entropy`（256 × vocab 的 exp/log）
- 两个 argmax + `multinomial_sample`（各 256 × vocab）

每个去噪步 ≈ 256 × 256k × 4 ≈ **2.6 亿次浮点操作，全在 CPU 单线程**，且夹在两次 GPU
提交之间，GPU 只能干等。48 步 × 这套 = wall-clock 被 host 拖垮。

**优化方向（预计 10–20× wall-clock 提升）：**

1. 把 argmax / entropy / 采样**下沉到 GPU**（一个 kernel 直接返回每位置的 argmax + entropy，
   而不是把整个 f32 logits 搬回 host）。
2. 退一步：至少用 OpenMP 并行化这些 per-position 循环，并与下一次 GPU 提交 **overlap**
   （异步 infer + pinned buffer）。
3. 不要为 256 个位置全量物化 f32 logits，只算需要的统计量。

---

## 4. 瓶颈 2：MoE 专家 GEMM 占 GPU 58.5%，效率仅 ~5%

执行序列证实了 MoE 结构（每个 MoE 层固定 3 个 `grouped_micro_gemm`，
gap 分簇得到 **1400 个 size=3 的簇**）：

```
moe_gather → grouped_micro_gemm(gate 3556us) → grouped_micro_gemm(up 3462us)
→ moe_3gemm_swiglu_fuse_prefill → grouped_micro_gemm(down 3798us) → moe_scatter
```

- **1440 次 decoder MoE 调用** = 48 去噪步 × 30 层（+30 encoder = 1470，
  与 `moe_router` 计数 1470 完全吻合）。
- `grouped_micro_gemm` 耗时极均匀：4320 次，min/median/p99/max =
  1210 / 2631 / 3346 / 4077 us，99% 落在 2–5 ms 区间。
- 估算：每 MoE 层 ~24 GFLOP / ~10.5 ms ≈ **2.3 TFLOP/s**，
  B390 的 INT4/fp16 峰值远超 50 TFLOP/s → **只用到约 5%**。
- 原因：256 token、top_k=8 → 2048 个 token-expert 分配散到 128 个专家，
  **每专家平均仅 ~16 token**。GEMM 太小，填不满 GPU，退化成访存受限的 `micro_gemm` 小核。

**优化方向：**

1. **融合 gate+up**：二者共享同一输入，现在是两个 `[2816→704]` 独立 grouped GEMM。
   合成一个 `[2816→1408]`，kernel 启动减半、算术强度翻倍。
2. 针对小 per-expert token 数换更合适的 grouped GEMM tiling（当前 `micro_gemm` 明显不适配）。

---

## 5. 瓶颈 3：48 步去噪 = 整模型重算 48 遍（成本乘数）

decoder 的开销 = 48 × 普通前向，是上面所有成本的乘数。

**优化方向（算法层）：** 让 early-stopping / entropy 接受率真正生效后步数会大幅下降 —— 但注意
目前正卡在 **NaN logits**（`mean_entropy=0.0000`、argmax 全 0），根本没在正常收敛。

> **先修 NaN，再谈减步**，否则减步只是更快地产出空输出。
> 每省 1 步 ≈ 省 30 个 MoE 层 ≈ 0.24 s GPU + 大量 host 时间。
> NaN 调试见 session 进展与 sample 中新增的 `nan=/inf=/max_logit=` 诊断。

---

## 6. 瓶颈 4：小 kernel 海（eltwise 14.8% + reorder 6.2%）

17,568 个 eltwise + 13,010 个 reorder/permute，单个都很小，
**启动开销 + 布局转换占了 GPU 忙碌的 21%**。reorder 纯粹是相邻 kernel 之间 layout 不匹配
产生的浪费。

**优化方向：** 图层融合（activation + eltwise + reduce 合并）、对齐相邻 kernel 的 tensor
layout 消除 reorder。

---

## 7. 优化优先级

| 优先级 | 项 | 预期收益 |
|---|---|---|
| **P0** | 修 host-bound 空转（logits 后处理并行化/下沉 GPU + 异步 overlap） | 砍 ~400 s wall-clock，10–20× |
| **P1** | 融合 MoE gate+up GEMM | 砍 GPU 计算最大头 |
| **P2** | 修 NaN 后减少去噪步数 | 每步 ≈ 0.24 s GPU + 大量 host |
| **P3** | 图融合消除小 kernel / reorder | 长尾 ~21% GPU 忙碌 |

---

## 8. 复现方法

```bash
# trace 文件：clintercept_trace.json（CLIntercept Chrome Trace 格式）
# 功能区聚合（区分 device kernel vs host API，按 thread_name 含 GPU 的 OOQ/IOQ 队列判定）：
python3 - <<'PY'
import json
from collections import defaultdict
data=json.load(open('clintercept_trace.json'))
tname={e['tid']:e['args']['name'] for e in data
       if e.get('ph')=='M' and e.get('name')=='thread_name'}
is_dev=lambda tid:'GPU' in tname.get(tid,'')
evs=[e for e in data if e.get('ph')=='X' and is_dev(e['tid'])]
evs.sort(key=lambda e:e['ts'])
span=(evs[-1]['ts']+evs[-1]['dur']-evs[0]['ts'])/1e6
busy=sum(e['dur'] for e in evs)/1e6
print(f"span={span:.1f}s busy={busy:.1f}s util={100*busy/span:.1f}%")
agg=defaultdict(lambda:[0,0.0])
for e in evs: agg[e['name']][0]+=1; agg[e['name']][1]+=e['dur']
for n,(c,d) in sorted(agg.items(),key=lambda kv:-kv[1][1])[:15]:
    print(f"{d/1000:9.1f}ms cnt={c:6d} avg={d/c:8.1f}us {n[:50]}")
PY
```

关键判定逻辑：
- **device kernel vs host API**：按 `thread_name` 是否含 `GPU` 且队列名以 `OOQ`/`IOQ`
  开头来区分（本 trace 中 host API 时长聚合为 0，全部时间在 device 队列）。
- **MoE 调用计数**：`grouped_micro_gemm` 总数 / 3 = 1440（每层 gate/up/down 三个 GEMM）。
- **去噪步数验证**：`moe_router` 计数 1470 = 30 encoder + 48 × 30 decoder 层。
