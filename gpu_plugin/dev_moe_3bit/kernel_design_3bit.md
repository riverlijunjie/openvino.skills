
About 3bit optimization for GPU kernels in OpenVINO:

Runtime kernel scheduler requirements:
  1. OneDNN gemm primitive is for 4bit/8bit quantization, we will implement new GPU 3bit gemm kernels without using OneDNN primitives(need oneDNN supported finally), so we need implement one kernel scheduler in primitive level to shedule different kernels for 3bit and 4bit/8bit quantization separately.

Weights is U2 and I4/U4 mixed quantization, for U2 optimization:
    weight data type: U2
    zp data type: U8
    scale data type: f16

OneDNN doesn't support I2/U2 weight quantization.

Kernel requirements:
  1. Prefill stage:
      - 3bit quantization for dense GEMM:
            activate data type: int8 (DQ), fp16 (less used)
            weight data type: U2 and I4/U4 mixed quantization
            output data type: f16
            computation: dpas(int8 x int4 -> int32 ->f16)
            kernel implement options(preferred option B): 
                option A: rewrite I2/U2 gemm kernel:  opencl/dpas_extension or cm
                option B: insert extra kernel to convert U2 to U4 before feeding into existing 4bit gemm kernel, we can reuse the existing gemm kernel but need allocate extra buffer for U4 expansion
      - 3bit quantization for MoE:
            activate data type: int8 (DQ), fp16 (less used)
            weight data type: U2 and I4/U4 mixed quantization
            output data type: f16
            computation: dpas(int8 x int8 -> int32 ->f16)
            kernel implement options(preferred option B): 
                option A: rewrite I2/U2 gemm kernel:  opencl/dpas_extension or cm
                option B: insert extra kernel to convert U2 to U4 before feeding into existing groupe_gemm, but need allocate extra buffer for U4 expansion
  2. Decode stage:
      - 3bit quantization for dense GEMV:
            activate data type: f16 (non-DQ)
            weight data type: U2 and I4/U4 mixed quantization
            output data type: f16
            computation: fma(f16 x f16 -> f32 -> f16)
            kernel implement options(preferred option A): 
                option A: rewrite I2/U2 gemm kernel:  opencl or cm
                option B: insert extra kernel to convert U2 to U4 before feeding into existing gemm kernel, we can reuse the existing gemm kernel but need allocate extra buffer for U4 expansion
      - 3bit quantization for MoE:
            activate data type: f16 (non-DQ)
            weight data type: U2 and I4/U4 mixed quantization
            output data type: f16
            computation: fma(f16 x f16 -> f32 -> f16)
            kernel implement: 
                option A: moe opencl enhancement(prefered)

Kernel optimization tips:

    GEMM optimization (dense prefill, compute-bound):
        - On the GPU we do NOT want to merely perform multiple kernels
        - Rectangular sub-group tiles to reuse maximally B / amortize B quantization overhead
        - Calculate B scales via optimized absmax reduction kernels, absmax reduction either as upfront kernel or fuse into GEMM microkernel
        - Bias addition and post-op (e.g. SiLU) fusion in the GEMM via epilogues
        - Use DPAS/XMX instructions for fused matrix multiply-accumulate:
            * dpas(int8 x int8 → int32 → f16) for 3-bit prefill (U2 weights expanded to int8)
            * dpas(int8 x int4 → int32 → f16) for dense 3-bit with int8 activation
        - U2 weight unpacking before DPAS:
            * 4 U2 values packed per byte; unpack via shift+mask: val_i = (byte >> (2*i)) & 0x3
            * Use vectorized uchar16/uchar8 block reads then bitfield extract in registers
        - Tiling strategy (target: 80% of hardware roofline):
            * Tile sizes: M=16~32, N=16~32, K=16~32 aligned to subgroup_size (16 or 32)
            * Use SLM (32 KB per Xe Core) for B tile buffering to reduce HBM round-trips
            * Double-buffer A and B tiles: load tile N+1 while computing tile N
        - Register management (256 regs × 256 bytes/EU on LNL XE2):
            * Monitor GRF pressure carefully: 3-bit dequant adds ~4-6 GRF vs INT4 path
            * Avoid register spilling; spill to SLM if needed, not to global memory
        - Minimize divergent branches inside inner K-loop; move scale/ZP load outside K-loop
        - Unroll K-loop by 2× or 4× to increase instruction-level parallelism

    GEMV optimization (dense decode, memory-bound):
        - GEMV is purely memory-bound (~4 FLOPs/byte for 2-bit INT); focus on bandwidth, NOT ALU
        - Target: ≥85% of memory roofline (e.g., LNL: 100 GB/s, BMG: 456 GB/s)
        - Computation: fma(f16 × f16 → f32 → f16) after dequant
        - Work-group layout:
            * subgroup_size = 32 (XE2+) or 16 (older architectures)
            * Each subgroup processes one N-block of elements (N-parallel, NOT K-parallel)
            * N_BLOCK=4, SUBGROUP_NUM=8 per workgroup → 256 threads per workgroup
            * lws = {1, SUBGROUP_SIZE, SUBGROUP_NUM}, gws = {1, SUBGROUP_SIZE, N/N_BLOCK}
        - U2 weight loading and dequant:
            * Block-read uchar16 chunks (16 bytes = 64 U2 values) with intel_sub_group_block_read
            * Unpack 4 U2 vals/byte: shift+mask in register, no SLM needed for dequant
            * Dequant formula: w_f16 = (cast_f16(u2_val) - zp) * scale
              where zp is u8 per group, scale is f16 per group
            * Load scale/ZP once per group (shared across SUBGROUP_SIZE lanes via broadcast)
        - Double-buffer weight tiles: issue next block-read while computing current block
        - Do NOT fuse gate+up into single GEMV kernel:
            * Fused variant doubles register pressure (from ~10-12 to ~20-24 GRFs)
            * Occupancy drop outweighs ALU savings for memory-bound kernels
            * Cache thrashing from accessing two disjoint weight regions simultaneously
            * Keep gate and up as separate kernel dispatches
        - FMA accumulation pattern:
            * Accumulate in f32 to avoid precision loss, convert to f16 at final write
            * Use sub_group_reduce_add for the K-dimension partial sums across lanes
        - Subgroup-level K-reduction:
            * Each lane handles K/SUBGROUP_SIZE elements, then reduce across subgroup
            * Prefer sub_group_reduce_add over manual shuffle trees for readability and correctness

    MoE prefill optimization (compute-bound at medium-large batch):
        - Regime: compute-bound above ~16 tokens/expert; use DPAS path
        - Policies:
            * Each experts in the same layer maybe have different weights quantization format (U2 or I4/U4),
              we should support them in the same kernel by separate load and dequant path for U2 and I4/U4 weights,
              then combine them together as the effective weight for DPAS computation.
            * Each expert may have different token number, we should keep thread workload balanced by assigning experts
              with similar token number to the same subgroup when possible, but workload imbalance is inevitable at
              low token number (e.g. <16 tokens/expert) and we should accept some imbalance to avoid excessive kernel
              complexity for perfectly balanced scheduling.
        - Weight dequant strategy for micro gemm kernel:
            * U2 weights must be expanded to U4 for micro gemm(f16 × u4 → f32 -> f16);
            * fuse expansion into GEMM microkernel prologue using gemmstone API:
              load U2 tile → unpack to U4 in registers → feed directly to micro kernel

    MoE decode optimization (memory-bound, single token):
        - Regime: purely memory-bound (~4 FLOPs/byte for 3-bit GEMV)
          XE2 compute crossover ≈ 239 FLOPs/byte → 3-bit is 60× below crossover
          Only data-movement reductions matter; ALU improvements are irrelevant
        - Pipeline: softmax_topk → mlp_gate_up → mlp_down → mlp_reduce (4 separate kernels)
        - Dispatch geometry for gate_up / down kernels:
            * gws = {num_experts, SUBGROUP_SIZE, INTERMEDIATE_SIZE/N_BLOCK}
            * lws = {1, SUBGROUP_SIZE, SUBGROUP_NUM}
            * N_BLOCK=4, SUBGROUP_NUM=8, SUBGROUP_SIZE=32 (XE2) / 16 (older)
            * 256 threads/workgroup, each workgroup handles N_BLOCK=4 output elements
            * Subgroups handle different N-block ranges (N-parallel), NOT redundant K
        - U2 weight dequant in decode kernel:
            * Each subgroup lane loads a uchar16 block = 64 U2 values (32-byte aligned)
            * Unpack per lane: val = (chunk >> (2 * lane_idx)) & 0x3 for each U2 value
            * Apply group-level scale (f16) and ZP (u8): w_f16 = (u2 - zp) * scale
            * For U2+I4 mixed 3-bit: separate load paths for U2 and I4 sub-tensors,
              dequant each separately then combine: w_eff = (2*w_i4 + w_u2) * scale
        - ZP optimization (xg_sum):
            * For symmetric 3-bit (no ZP): skip xg_sum accumulation entirely
            * Guard with #if HAS_ZP / #else to compile out ZP code path at JIT time
            * JIT constant HAS_ZP: 0=symmetric (no ZP code), 1=asymmetric
        - Do NOT fuse gate+up GEMV into a single kernel (empirically slower):
            * Register pressure doubles (~10-12 → ~20-24 GRFs)
            * Occupancy drops, hiding less memory latency
            * Cache thrashing: gate_w and up_w are disjoint memory regions
        - Fuse mlp_reduce into mlp_down kernel (low complexity, ~1ms gain over 48 layers):
            * After down GEMV, routing weight multiplication and expert accumulation can be
              done in the same workgroup without writing intermediate results to global memory
        - Shared expert fusion (Qwen3.5-style models):
            * Treat shared expert as extra workgroup in dim-0 (expert index = MAX_TOPK)
            * Scalar gate for shared expert: workgroup-wide parallel dot-product reduction
              (256 threads → reduce_add → sigmoid), not single-thread sequential
            * JIT constant SHARED_EXPERT_ENABLE=1 to compile in extra workgroup handling
        - SLM usage in decode:
            * Use SLM to preprocess activation x: interleave even/odd elements for 4-bit
              compatibility; direct copy for 8-bit/f16
            * SLM broadcast: one subgroup loads scale/ZP, others read via SLM broadcast
              to avoid redundant global memory loads of quantization parameters
        - Double-buffer weight tile loading:
            * Prefetch next weight tile while computing current tile's dequant+FMA
            * Hides ~50% of HBM latency at cost of 2× working-set registers for weight tiles
