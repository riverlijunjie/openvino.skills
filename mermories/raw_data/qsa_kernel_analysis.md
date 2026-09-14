# QSA CM Kernel 脚本分析 - 性能路径优化

## 脚本分类与依赖关系

### A. 独立单层次Kernel测试（基础建置层）
这些脚本测试单个kernel，各自独立：
- `test_q0_kv_cache_update.py` - Q0: KV缓存更新（paged scatter）
- `test_q1_prepare.py` - Q1: 投影+归一化+RoPE
- `test_q2_score_partition.py` - Q2: 分区评分（decode）
- `test_q2_score_tile_dpas.py` - Q2: 瓦片评分（prefill，DPAS）
- `test_q2_score_topk_fused.py` - Q2: 融合评分+top-k
- `test_q2_topk_finalization.py` - Q2: Top-k最终化（基础radix）
- `test_q3_sparse_attention.py` - Q3: 稀疏注意力（标量，生产decode kernel）
- `test_q3_sparse_attention_dpas.py` - Q3: 稀疏注意力（DPAS prefill）
- `test_gr_gated_residual.py` - 门控残差分支kernel

### B. 验证脚本（精确路径验证，无性能基准）
- `validate_q1_dpas.py` - Q1: DPAS路径 vs 标量基准
- `validate_q2_chunked.py` - Q2: 行分块计划（CPU规划仅）
- `validate_q2_cooperative.py` - Q2: 合作top-k finalizer
- `validate_q3_dpas.py` - Q3: 4种DPAS优化阶段（可选benchmark）
- `validate_q3_split.py` - Q3: 分区实现（可选benchmark）
- `validate_q3_gqa.py` - Q3: GQA对象分派（可选benchmark）
- `validate_topk_fast.py` - Q2: 新fast/fast-wg16 finalizer
- `validate_dense_bypass.py` - Q2: 密集旁路优化
- `test_topk_finalizer_options.py` - CPU-only: finalizer合约验证

### C. 集成管道测试（完整端到端）
- `test_qsa_pipeline.py` - 完整Q0→Q1→Q2→Q3流程（含状态管理）
- `test_q3_gqa_options.py` - GQA dispatching契约（CPU-only，无GPU）

### D. 完整性能基准测试（生产测试）
**PTL (B390 iGPU):**
- `benchmark_ptl_optimized.py` - PTL A/B baseline vs optimized（full pipeline）
- `benchmark_ptl_q3_variants.py` - PTL Q3: baseline/head-major/head-major-meta
- `benchmark_topk_pipeline.py` - PTL Q2 finalizer对比（legacy/fast/fast-wg16）

**BMG (B580):**
- `benchmark_long_context.py` - BMG完整QSA（8种大小×2相位×2缓存）
- `profile_q3_dpas.py` - Q3 DPAS调优敏感性分析

### E. 离线分析与审计（只读，无GPU）
**报告生成:**
- `analyze_long_context_roofline.py` - BMG roofline报告（28条记录）
- `analyze_ptl_roofline.py` - PTL roofline报告（同方法论，不同硬件常数）
- `analyze_fast_topk_roofline.py` - fast finalizer roofline（14条新数据）
- `analyze_topk_pipeline.py` - PTL top-k管道审计（无roofline）

**历史审计:**
- `audit_long_context_history.py` - 重现原始SHA256（反向源delta）

**测试的测试:**
- `test_analyze_long_context_roofline.py` - roofline分析回归
- `test_analyze_fast_topk_roofline.py` - fast finalizer roofline验证
- `test_analyze_ptl_optimization.py` - PTL A/B日志完整性验证

## 性能关键路径识别

### 生产路径（必须保留）
**Prefill:**
1. Q0: `test_q0_kv_cache_update.py`（必须）
2. Q1: `test_q1_prepare.py` + `validate_q1_dpas.py`（DPAS路径是最优）
3. Q2评分: `test_q2_score_tile_dpas.py`（DPAS prefill）
4. Q2 top-k: `validate_topk_fast.py`（fast-wg16-cached是最优）
5. Q3: `test_q3_sparse_attention_dpas.py` + `validate_q3_dpas.py`（最优DPAS）

**Decode:**
1. Q0: `test_q0_kv_cache_update.py`（必须）
2. Q1: `test_q1_prepare.py`（标量基准）
3. Q2评分: `test_q2_score_partition.py`（decode路径）
4. Q2 top-k: `validate_topk_fast.py`（fast-wg16-cached是最优）
5. Q3: `test_q3_sparse_attention.py`（标量是生产decode kernel）

### 性能测试基准
**完整集成:**
- ✅ `benchmark_ptl_optimized.py` - PTL A/B（PTL是生产target）
- ✅ `benchmark_long_context.py` - BMG完整流程（技术验证）
- ✅ `benchmark_topk_pipeline.py` - top-k对比（可选但useful）

**单层kernel:**
- ✅ 所有 test_q*.py（--iters + roofline）
- ⚠️ `profile_q3_dpas.py`（调优分析，非回归）

## 冗余/非必要脚本分类

### 1. 测试的测试（可删除）
**删除风险: 极低** - 这些是单元测试，不是生产路径
- `test_analyze_long_context_roofline.py` - 分析脚本测试
- `test_analyze_fast_topk_roofline.py` - 分析脚本测试
- `test_analyze_ptl_optimization.py` - PTL日志审计测试
- `test_q3_gqa_options.py` - CPU-only GQA契约验证
- `test_topk_finalizer_options.py` - CPU-only finalizer合约
→ **可删除理由**: 仅为验证分析代码正确性，不覆盖实际kernel性能

### 2. 历史维护与审计脚本（可删除）
**删除风险: 低** - 支持历史再现，非生产
- `audit_long_context_history.py` - 反向delta重现原始测试
- `analyze_topk_pipeline.py` - PTL top-k审计（不含roofline）
→ **可删除理由**: 支持历史验证，无新性能基准；PTL roofline已有 analyze_ptl_roofline.py

### 3. 超范围验证脚本（可删除）
**删除风险: 中** - 这些验证非关键路径或已在集成中覆盖
- `validate_q3_gqa.py` - GQA方案对比（已由 benchmark_ptl_q3_variants.py覆盖）
- `validate_q3_split.py` - Q3分区实现（已由集成pipeline覆盖）
- `validate_q2_cooperative.py` - 合作top-k（已由 validate_topk_fast.py覆盖）
- `validate_dense_bypass.py` - 密集旁路（Q2设计验证，非性能关键）
→ **可删除理由**: 这些路径已整合进生产benchmark；单独运行无新性能数据

### 4. 基础kernel测试但性能路径重复（可优化保留）
**删除风险: 中-高** - 性能基准是重复的，但保留用于单kernel回归
选择保留**必要最小集**：
- ✅ `test_q0_kv_cache_update.py` - 必须（Q0无变体）
- ✅ `test_q1_prepare.py` - 保留（基础投影基准）
- ⚠️ `test_q2_score_partition.py` - decode独占但性能已在 benchmark_long_context.py
- ⚠️ `test_q2_score_tile_dpas.py` - prefill独占但性能已在 benchmark_long_context.py
- ❌ `test_q2_score_topk_fused.py` - **可删除**（融合top-k非生产路径）
- ❌ `test_q2_topk_finalization.py` - **可删除**（基础radix已由fast替代）
- ✅ `test_q3_sparse_attention.py` - 保留（生产decode kernel）
- ✅ `test_q3_sparse_attention_dpas.py` - 保留（生产prefill kernel）
- ⚠️ `test_gr_gated_residual.py` - 保留（独立kernel，性能关键）

### 5. 分析/配置脚本（可精简）
**删除风险: 低** - 支持结果分析
- ✅ `analyze_long_context_roofline.py` - BMG报告（保留）
- ✅ `analyze_ptl_roofline.py` - PTL报告（保留）
- ⚠️ `analyze_fast_topk_roofline.py` - fast finalizer报告（已整合进PTL?检查）
- ⚠️ `profile_q3_dpas.py` - 调优分析（非回归，可删除）

## 推荐最小性能回归集

### 保留用于生产验证（必须）
```
test_q0_kv_cache_update.py        # Q0基准
test_q1_prepare.py                # Q1基准
test_q3_sparse_attention.py       # Q3 decode生产
test_q3_sparse_attention_dpas.py  # Q3 prefill生产
validate_q1_dpas.py               # DPAS投影验证
validate_topk_fast.py             # 最优top-k验证
benchmark_ptl_optimized.py        # PTL A/B（生产target）
benchmark_long_context.py         # BMG完整集成
analyze_ptl_roofline.py           # PTL结果报告
analyze_long_context_roofline.py  # BMG结果报告
```

### 可删除（无性能回归价值）
```
test_analyze_*.py                 # 分析脚本测试
test_topk_finalizer_options.py    # CPU-only合约
test_q3_gqa_options.py            # CPU-only GQA
test_q2_score_topk_fused.py       # 非生产路径
test_q2_topk_finalization.py      # 已被fast替代
validate_q3_gqa.py                # 已由benchmark_ptl_q3_variants.py覆盖
validate_q3_split.py              # 集成已覆盖
validate_q2_cooperative.py        # 已由validate_topk_fast.py覆盖
validate_dense_bypass.py          # 设计验证，非性能关键
validate_q2_chunked.py            # CPU规划仅
audit_long_context_history.py     # 历史维护
analyze_topk_pipeline.py          # 已由analyze_ptl_roofline.py覆盖
profile_q3_dpas.py                # 调优分析非回归
benchmark_topk_pipeline.py        # 可选（已由集成覆盖）
benchmark_ptl_q3_variants.py      # 已集成进benchmark_ptl_optimized.py
```

### 保留但可考虑清理（现有性能数据但低优先级）
```
test_q2_score_partition.py        # decode评分基准（已在benchmark_long_context.py）
test_q2_score_tile_dpas.py        # prefill评分基准（已在benchmark_long_context.py）
test_gr_gated_residual.py         # GR独立kernel（可保留）
analyze_fast_topk_roofline.py     # fast finalizer报告（检查是否已整合）
```
