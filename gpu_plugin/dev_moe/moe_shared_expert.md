# Qwen 3.5 MoE Shared Expert Optimization

## Overview
Optimized the Qwen 3.5 MoE architecture by fusing the "Shared Expert" computation into the main `MOE3GemmFusedCompressed` operator. This eliminates the need for separate MatMul/activation operations for shared experts, reducing kernel launch overhead and memory traffic.

## Implementation Details

### 1. OpenVINO Core Operator Update
*   **Target Op**: `MOE3GemmFusedCompressed` (and corresponding GPU primitive).
*   **Change**: Extended the operator signature to accept optional inputs (indices 11-20) specifically for shared expert parameters.
*   **New Inputs**:
    *   `shared_gate_weight/scale/zp`: Shared Expert Gate Projection.
    *   `shared_up_weight/scale/zp`: Shared Expert Up Projection.
    *   `shared_down_weight/scale/zp`: Shared Expert Down Projection.
    *   `shared_gate_gate_weight`: The sigmoid gate weight for the shared expert output.

### 2. GenAI Library Integration
*   **File**: `openvino.genai/src/cpp/src/modeling/models/qwen3_5/modeling_qwen3_5_moe.cpp`
*   **Logic Change**:
    *   In the fused path (`routed_fused`), we now prepare the shared expert weights.
    *   Since shared experts in Qwen 3.5 are typically uncompressed (FP16/FP32), we generate dummy scales (1.0) and zero-points (0.0) to match the compressed operator interface.
    *   Weights are reshaped to 4D tensors (e.g., `[1, hidden, 1, 1]`) to comply with the blocked format expected by the kernel.
    *   The `forward` pass was simplified to rely on the fused operator for the entire MoE block (Routed + Shared) when possible.

### 3. Key Benefits
*   **Performance**: Single fused kernel execution for both routed and shared components.
*   **Memory**: Reduced intermediate memory allocations for shared expert activations.
*   **Uniformity**: Consistent interface for compressed and uncompressed shared expert handling.

## Related Files

### Core & GPU Plugin
*   `openvino/src/core/dev_api/openvino/op/moe_3gemm_fused_compressed.hpp`: Op definition.
*   `openvino/src/plugins/intel_gpu/include/intel_gpu/primitives/moe_3gemm_fused_compressed.hpp`: GPU primitive definition.

### GenAI Modeling
*   `openvino.genai/src/cpp/src/modeling/models/qwen3_5/modeling_qwen3_5_moe.cpp`: Model implementation.
*   `openvino.genai/src/cpp/src/modeling/ops/ops.cpp`: Op construction helper.
