# Shared Expert Optimization

## Overview

In architectures like Qwen3.5 MoE, the MoE block features a **Shared Expert** — a parallel feed-forward network whose output is added directly to the routed sparse experts' output.

## Operator Extension

`MOE3GemmFusedCompressed` extended with optional inputs (indices 11-20):
- `shared_gate_weight/scale/zp`: Shared Expert Gate Projection
- `shared_up_weight/scale/zp`: Shared Expert Up Projection
- `shared_down_weight/scale/zp`: Shared Expert Down Projection
- `shared_gate_gate_weight`: Sigmoid gate weight for shared expert output

## Decode Path (Fused)

- Shared expert treated as expert `MAX_TOPK` (extra workgroup in dim-0)
- Scalar gate: workgroup-wide parallel dot-product reduction (256 threads), sigmoid, writes `routing_weights[MAX_TOPK]`
- Same GEMV functions used for both sparse and shared experts

## Prefill Path (Not Yet Fused)

- `execute_shared_expert()` runs 3 separate oneDNN matmul calls after main MoE
- Serialized with preceding kernels by oneDNN stream
- Future: incorporate into grouped dispatch

## GenAI Integration

In `openvino.genai/src/cpp/src/modeling/models/qwen3_5/modeling_qwen3_5_moe.cpp`:
- Fused path prepares shared expert weights
- Uncompressed shared experts (FP16/FP32) use dummy scales (1.0) and zero-points (0.0) to match compressed interface
- Weights reshaped to 4D tensors for blocked format expected by kernel

## Graph Transformation Fix

**Problem**: Pattern matcher collision — standard MOE matcher greedily consumed inner MOE node before shared expert matcher matched `Add(MOE, SharedExpert)`.

**Solution**:
1. Rejection predicate `!ov::is_type<ov::op::v1::Add>` on standard matcher
2. Context-aware `replace_node`:
   ```cpp
   if (has_shared_expert)
       ov::replace_node(root_node, moe_compressed);  // Replace outer Add
   else
       ov::replace_node(moe, moe_compressed);         // Replace standard MoE
   ```
3. Test infrastructure updated for 22-parameter shared expert layout (vs 12 for standard)
