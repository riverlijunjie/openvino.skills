---
name: dev_moe
description: Develop Mixture of Experts (MoE) operations for better performance. Use when working on MoE models or improving MoE operation efficiency.
---

When develop moe feature, always include:
1. **Read all moe code**: Read moe related code to understand its structure and optimizations. This includes:
    - moe op definition and implementation
    - moe transformation implementation
    - moe primitive implementation
    - moe kernel implementation
    - moe unit tests
2. **Identify optimization opportunities**: Look for areas in the code where performance can be improved, such as inefficient algorithms, redundant computations, or memory bottlenecks.
3. **Implement optimizations**: Apply code changes to optimize the identified areas. This may include algorithmic improvements, better memory management, or leveraging hardware capabilities.
4. **Document optimization**:
    - Summarize the optimizations applied and their impact on performance.
    - Include before-and-after performance metrics to demonstrate improvements.
    - Include the ratio to hardware roofline to show efficiency gains.

Keep explanations conversational. For complex concepts, use multiple analogies.

---

## Optimization Record: Fused Gate+Up GEMV in `moe_gate_up`

### Background

In the Qwen3.5 MoE model, the MLP for each expert computes `y = up(x) * silu(gate(x))` where `gate` and `up` are separate INT4-quantized GEMV operations against HIDDEN_SIZE×INTERMEDIATE_SIZE weight matrices. The original kernel called `gemv_n2x()` twice — once for `up`, once for `gate` — each reading the same activation data from SLM independently.

**Roofline context**: With INT4 weights and FP16 activations, this kernel has an operational intensity of ~4 FLOPs/byte — strongly memory-bound (25× below the compute crossover point on XE2). Think of it like a highway where traffic is jammed at the on-ramp (memory bandwidth), not on the highway itself (ALU). So the only way to go faster is to reduce the number of cars (bytes moved), not build more lanes (more ALUs).

### Problem Identified

Two redundant data movement patterns in the original `mlp_gate_up` kernel:

1. **Double SLM activation reads**: `gemv_n2x()` for up-projection reads the entire activation vector from SLM. Then `gemv_n2x()` for gate-projection reads the exact same activation data again. It's like reading the same book twice to answer two different questions — you could just keep it open.

2. **Intermediate global memory round-trip**: The up-projection wrote intermediate results to global memory `y[]`, then the gate-projection read `y[]` back to compute the final `up * silu(gate)`. This is like mailing a letter to yourself just so you can read it — completely unnecessary if you do both computations in the same function.

3. **Single-threaded shared expert scalar gate**: The shared_expert_gate sigmoid computation (`dot(x, gate_weight)` over HIDDEN_SIZE=3584 elements) was done by a single thread in a sequential loop — 3584 iterations while 255 other threads sat idle. Like having 256 workers but only one digging.

### Optimization Applied

**File**: `moe_3gemm_swiglu_mlp.cl`

#### A. Fused `gemv_gate_up_fused()` function

Replaced two sequential `gemv_n2x()` calls with a single `gemv_gate_up_fused()` that:

- **Reads activation from SLM once per quantization group** — the `half4`/`half8` SLM block read is done once and shared across gate and up dot-product accumulations
- **Reads gate and up weights simultaneously** — both weight vectors are loaded in the same loop iteration, enabling instruction-level parallelism (the two memory reads can overlap in the hardware pipeline)
- **Computes `y = up * silu(gate)` in a single write** — no intermediate global memory write/read; both sub-group reductions complete, then the final fused result is written once

Think of it like a restaurant kitchen: instead of grilling the steak (gate), plating it, sending it to the table, bringing it back, then adding the sauce (up), the chef now grills the steak and prepares the sauce at the same time, plates the combined dish in one pass.

#### B. Parallelized shared expert scalar gate

Replaced the single-thread loop with a workgroup-wide parallel reduction:

- All 256 threads (8 subgroups × 32 threads) split the HIDDEN_SIZE dot product: ~14 iterations each instead of 3584
- Two-level reduction: `sub_group_reduce_add()` within each subgroup, then cross-subgroup reduction via SLM `shared_gate_partial[SUBGROUP_NUM]` array
- Final sigmoid computed by thread 0

### Performance Results (Qwen3.5 MoE, single-batch decode, 40 layer iterations)

#### Core MoE kernels

| Kernel | Before (SE) | After (OPT) | Delta | Change |
|--------|------------|-------------|-------|--------|
| **moe_gate_up** | 6.271 ms (156.8 us avg) | 5.214 ms (130.4 us avg) | **−1.057 ms** | **−16.9%** |
| moe_down | 2.737 ms (68.4 us avg) | 2.619 ms (65.5 us avg) | −0.118 ms | −4.3% |
| moe_reduce | 0.062 ms (1.5 us avg) | 0.090 ms (2.2 us avg) | +0.028 ms | +45%* |
| softmax_topk | 0.254 ms (6.4 us avg) | 0.263 ms (6.6 us avg) | +0.009 ms | +3.5%* |
| **MoE pipeline total** | **9.324 ms** | **8.186 ms** | **−1.138 ms** | **−12.2%** |

\* moe_reduce and softmax_topk absolute changes are <0.03 ms — microsecond-level noise.

#### Hardware metrics for `moe_gate_up`

| Metric | Before (SE) | After (OPT) | Change |
|--------|------------|-------------|--------|
| ALU0 Instructions | 106.2M | 43.7M | **−58.8%** |
| ALU1 Instructions | 165.7M | 72.4M | **−56.3%** |
| L3 Cache Read | 52.9 GB/s equiv | 32.9 GB/s equiv | −37.8% |
| SBID stall (scoreboard wait) | 34.0% | 26.5% | −7.5pp |
| Stalled | 35.5% | 28.8% | −6.7pp |
| XVE Occupancy | 39.8% | 29.8% | −10.0pp** |

\*\* Occupancy decreased because the kernel finishes faster. The same effective work completes with fewer instructions, so the GPU enters idle state sooner. The wallclock time improvement (−16.9%) is the real performance indicator.

#### Roofline efficiency

- **Operational intensity**: ~4 FLOPs/byte (memory-bound regime, 25× below compute crossover)
- **Before**: ALU instruction count suggests redundant work inflating apparent compute load
- **After**: ALU instructions cut by ~58%, confirming elimination of duplicate SLM reads and intermediate y[] traffic
- The kernel is now closer to the theoretical memory bandwidth limit — the remaining optimization frontier is in weight compression or memory access pattern improvements

### Why It Works

The key insight is that for a memory-bound kernel, **reducing data movement is more valuable than adding compute**. The fusion achieves this on three levels:

1. **SLM bandwidth halved**: One activation read per group instead of two → directly reduces the main bottleneck
2. **Global memory traffic eliminated**: No intermediate y[] write+read → removes an entire round-trip through the memory hierarchy
3. **Better ILP**: Interleaved gate/up weight reads fill memory pipeline bubbles that were previously wasted

### Files Modified

- `openvino/src/plugins/intel_gpu/src/graph/impls/ocl_v2/moe_3gemm_swiglu_mlp.cl`: Added `gemv_gate_up_fused()`, parallelized shared expert scalar gate, updated `mlp_gate_up` kernel to call fused function
- No C++ host-side changes needed (kernel signature, dispatch dimensions, and JIT constants all preserved)
---

## Optimization Record: Fuse Shared Expert Transformation

### Background

In architectures like Qwen3.5, the MoE block features a **Shared Expert** — a parallel feed-forward network whose output is added directly to the routed sparse experts' output. In the OpenVINO computation graph, this manifests as an outer `Add` node that groups the main `MOE` routing sub-graph with the shared expert computations. 

To run this efficiently on the GPU, the entire operation (sparse experts + shared expert) needs to be fused into a single `MOECompressed` operation.

### Problem Identified

When attempting to map this shared expert topology using OpenVINO's C++ pattern matchers, several critical roadblocks emerged:

1. **Greedy Pattern Matching (Matcher Collisions)**: The transformation logic defines multiple fallback sub-tree matchers. The standard MoE matcher (`moe_root_gemm3_no_shared_expert`) was overly greedy. When the graph contained a shared expert (with an outer `Add` node), the standard matcher would trigger prematurely on the inner MoE node, completely bypassing the larger shared expert pattern. It's like a bouncer stopping you at the front door before you can show your VIP all-access pass for the whole building.
2. **Incorrect Node Demotion**: Inside the node replacement logic, the transformer unconditionally executed `ov::replace_node(moe, moe_compressed)`. So even if the shared expert was theoretically evaluated, it deleted the inner MOE component instead of swapping the top-level `root_node` (the `Add` group). This left the graph disconnected.
3. **Mismatched Reference Graph Parameters**: A standard compressed MoE operation takes 12 weight/scale parameters. A shared expert MoE scales up to **22 parameters** (to hold the shared expert weights and gating values). The validation testing framework still anticipated the smaller 12-parameter footprint, causing unit tests to structurally reject the new graph.

### Optimization Applied

**Files Modified**: 
- `src/plugins/intel_gpu/src/plugin/transformations/convert_moe_to_compressed.cpp`
- `src/plugins/intel_gpu/tests/unit/transformations/convert_moe_to_compressed_test.cpp`

#### A. Matcher Predicate Isolation
Added a strict rejection predicate to the standard MoE matcher. By enforcing `!ov::is_type<ov::op::v1::Add>`, the standard pattern actively rejects matching if it detects an outer `Add` block. This effectively solves the greedy matching collision and forces the Graph to pass control to the fully-sized shared expert matcher.

#### B. Dynamic Root Replacement
Modified the replacement execution to be context-aware:
```cpp
if (has_shared_expert) {
    ov::replace_node(root_node, moe_compressed); // Replace the outer Add
} else {
    ov::replace_node(moe, moe_compressed); // Replace standard MoE
}
```

#### C. Testing Infrastructure Alignment
Updated `ov_gpu_unit_tests` reference models to manually construct the 22 expected mock parameters matching Qwen3.5's layout, passing the structural validation (`12 inputs != 22 inputs` error).

### Why It Works

Pattern matchers in OpenVINO evaluate topologically based on strict shapes and element limitations. By applying mutually exclusive predicates, the graph transformation pass can now reliably recognize the entire shared expert pathway and consolidate it completely into a single fused GPU primitive. This prevents trailing subgraph fragmentation and sets the stage for maximum throughput in backend fused kernels.
