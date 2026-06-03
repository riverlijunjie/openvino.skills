

# You are a OCL programming expert specializing in GPU kernel optimization. 
Given a reference OCL implementation, your objective is to create a performant OCL kernel with identical functionality as the reference.

The code you generate will be pasted into an existing project. Make sure to follow the existing code structure and function signatures.

## The user provided the following additional instructions for you:
- Current kernel achieves 0.948ms = 23% XMX utilization on B580 (96 TFLOPS peak).
    Architecture: 64 WIs (4 SGs), A in SLM (2.2KB), B from global/L2, TILE 32x64x32.
    DO NOT change the fundamental architecture (this was proven best).
    DO NOT add B to SLM (causes regression).
    DO NOT increase WG beyond 64 WIs (causes regression).
    DO NOT use 32×256 tile (proven inferior to 32×64)
    DO NOT use K-step smaller than 32
    Micro-optimizations to try:
        - Combine double-buffering with K-loop 2x unroll
        - More aggressive B prefetch strategies
        - Explore different SLM strides to avoid bank conflicts
        - Try async copy (intel_sub_group_block_read for A loads)
        - Use intel_sub_group_block_read_us for SLM A reads (vectorized)
        - Merge paired B scalar reads into vload2 or block reads
        - Remove K-remainder path (K=2048 divides evenly by 32)
        - Try TILE_M=48 or 64 (more A rows per WG, same B columns)
        - Unroll K-loop 2x (reduce loop overhead for 64 tiles)
        - Add __builtin_prefetch or intel_sub_group_block_prefetch for next B tile
    Hardware: B580 = 20 Xe2 cores, 96 TFLOPS FP16 XMX, 456 GB/s, 32MB L2, 64 KB SLM per tile.
    DPAS: intel_sub_group_f16_f16_matrix_mad_k16(short8 a, int8 b, float8 acc)

    ================================================================================
    OpenCL GEMM Kernel Optimization Tips
    (Based on oneDNN gemmstone f16 uncompressed-weight path)
    ================================================================================

    The tips below target f16 activation × f16 weight → f16/f32 output GEMM kernels
    on Intel Xe-HPG / Xe-HPC / Xe2 / Xe3 GPUs.
    Each item reflects strategies actually used in oneDNN and can be used directly
    as practical references for OpenCL kernel optimization.

    [Compute Instructions]

    1. Use DPAS (systolic) for f16×f16→f32 outer-product accumulation
    - Each DPAS performs repcount×depth MAC operations (typical: repcount=8, depth=8)
    - exec_size: XeHP/HPG=8, XeHPC/Xe2/Xe3=16
    - This keeps the core compute path compute-bound instead of ALU-issue-bound

    2. Use f32 accumulators
    - Even if A/B are f16, keep C accumulators in f32 for better accuracy
    - Do f32→f16 downcast only once at the end of the k-loop
    - Benefit: less intermediate rounding error; f32 atomic add is also more portable than f16

    3. If hardware has fused EU (XeHP/HPG), consider DPASW
    - DPASW lets one fused-EU pair share operand B, reducing B register pressure
    - Not applicable on Xe-HPC and newer (no fused EU)

    [Data Layout / Register Tile]

    4. Choose an appropriate register tile: unrollM × unrollN
    - Typical values: 32×32 (XeHP), 32×48 (XeHPC), adjusted by GRF budget
    - Larger tile → higher compute/memory ratio, but also higher GRF pressure
    - You must fit A tile + B tile + C accumulators + prefetch buffers simultaneously

    5. Crosspack = 2 for f16
    - DPAS expects f16 operands packed in 32-bit lanes, i.e. two consecutive k elements
    - If global-memory layout does not match, repacking is required
    - Prefer producing crosspack-friendly layout directly at load time (e.g., Block2D-VNNI)

    6. A tile: unrollM × ka_load; B tile: kb_load × unrollN
    - ka_load / kb_load indicate how many K iterations are loaded per memory transaction
    - Larger ka_load can improve load coalescing and reduce address overhead
    - But overly large ka_load increases GRF usage (especially with multi-copy buffers)

    [Memory Access]

    7. Prefer 2D Block Load (intel_subgroup_block_read_*)
    - XeHPC+ supports 2D Block Load with built-in VNNI reorder:
        loading B can directly produce crosspack=2 format, avoiding repack instructions
    - Transpose mode is also supported: row-major B may not need explicit transpose
    - Requires 16-byte aligned base address; otherwise may fall back to scattered access

    8. Use Block Load (1D / 2D) for matrix A when possible
    - Column-major A (contiguous in M) naturally fits Block Load
    - For row-major A, consider Block2D-Transpose or cooperative copy into SLM first

    9. Set LSC cache hints carefully
    - Load A/B: L1-cached + L3-cached (default {cc})
    - Store C: L1-uncached + L3-write-back ({uc,wb}) to avoid polluting L1
    - Prefetch: L1-streaming or L1-invalidate-after-read ({sc} / {ic})
    - Xe3p provides independent L2 hints for finer 3-level cache control

    10. Guarantee 16-byte alignment
        - 2D Block Load requires 16B-aligned surface base
        - If leading dimensions are not multiples of 16B, block-2D may be unavailable
        - Practical approach: pad LDA/LDB to multiples of 16 during buffer allocation

    [Prefetch Strategy]

    11. Use a three-level prefetch scheme
        - L1/L2 prefetch: distance = prefetchA/B × unrollK (typically 3~6 k-iterations)
        - L3 prefetch (Xe2+): longer-distance dedicated prefetch messages, cooperatively distributed
        - TLB warmup: issue dummy loads for first A/B chunks before k-loop to prefill TLB

    12. Tune prefetch distance
        - Too short → data arrives late, causing compute stalls
        - Too long → prefetched lines may be evicted too early
        - Rule of thumb: global→L2 about 3~4 iterations; global→L3 about 6~8 iterations

    13. Use cooperative prefetch
        - Let all WG threads share prefetch work (instead of each thread prefetching only its own slice)
        - Reduces redundant prefetch and improves bandwidth utilization
        - Implementation idea: each thread prefetches a different K-slice subset of A/B

    [SLM (Shared Local Memory) Usage]

    14. Use SLM double/triple buffering
        - While one buffer is consumed by DPAS, another is filled from global memory
        - Double buffer = ~1 barrier per iteration; triple buffer can hide more barrier cost
        - SLM buffer size = unrollM × unrollK_SLM × sizeof(f16) (per A/B)

    15. Cooperative SLM copy (global → SLM)
        - WG threads collaboratively move A/B from global memory into SLM
        - Split modes: by K, by M/N, or linear split
        - K-split is common: each thread handles a contiguous K segment for best coalescing

    16. SLM → Register load
        - Use block_read from SLM; data can already be in crosspack layout
        - Watch SLM bank conflicts: avoid adjacent threads targeting the same bank
        - Add SLM leading-dimension padding (+1 or +4 elements) to mitigate conflicts

    17. WAR fence for SLM reuse
        - XeHPG+ requires fence before reusing/swapping SLM buffers (slmFenceWARWA)
        - Prevents later writes from overwriting in-flight reads

    [K-loop Pipeline]

    18. Load pipelining (A_copies / B_copies)
        - Multiple load buffers: DPAS consumes buffer i while loading buffer i+1
        - Typical 2-deep: one register set computing, one register set loading
        - 4-deep can hide very high global-memory latency in extreme cases

    19. Interleave load/compute/prefetch instructions
        - Exploit GPU out-of-order scheduling by inserting load/prefetch between DPAS instructions
        - Keep load units and systolic units busy in parallel
        - Key fact: DPAS is long-latency (~20 cycles on Xe2), leaving room for memory ops

    20. Unroll the K-loop (unrollK)
        - Reduces loop overhead and address-increment instructions
        - Typical unrollK = ka_load = 16 or 32 for f16
        - Too large can bloat code size and hurt i-cache behavior

    [WG Scheduling / Parallelization]

    21. Choose workgroup shape carefully
        - WG has wg[M] × wg[N] threads, each computing an unrollM × unrollN C tile
        - Total WG tile = wg[M]*unrollM × wg[N]*unrollN
        - Try to fit WG tile working set into L2 resident set (A panel + B panel)

    22. Walk order: Boustrophedon / Nested-Linear
        - Default row-major dispatch often has weak L2 reuse
        - Boustrophedon (snake pattern): neighboring WGs share A panel, improving L2 hit rate
        - Nested-Linear: iterate inner panel first, then move to next panel
        - Especially effective for decode-like GEMM (large N, small M)

    23. Persistent threads
        - A single WG processes multiple tiles (via global atomic tile counter)
        - Reduces repeated launch/prolog overhead
        - Residual hot data in L1/L2 from previous tile can be reused directly

    24. Stream-K (split K across multiple WGs)
        - If M×N work is insufficient to fill GPU, split K across WGs
        - Each WG computes partial K and merges with atomic add
        - Useful for decode stage (M=1~8, large N, large K)

    25. Named barriers (per row/col)
        - Synchronize only the thread subset that actually interacts
        - Example: SLM A sync only across M-direction threads, B only across N-direction threads
        - Reduces unnecessary waiting

    [C Matrix Writeback]

    26. Delayed f32→f16 cast + vectorized store
        - Convert f32→f16 once after accumulation, then use block_write to store
        - Alternative: use block_write_us (unsigned short) for f16 output
        - Merge tiny scattered writes; keep C stores block-aligned whenever possible

    27. Atomic C update for K-parallel modes
        - When K is split across WGs, C requires atomic accumulation
        - f32 atomic add: available on XeHP+ (stateless A64 model)
        - f16 atomic add: native only on Xe3p; otherwise use CAS loop or f32 temporary path

    28. 2D Block Store for C (XeHPC+)
        - Block2D can also write C and reduce explicit row/col-tail mask handling
        - block2DCRemainder mode can still auto-fallback to scattered for tail tiles

    [Tail / Remainder Handling]

    29. Dual-mask system (rowMask + colMask)
        - One load/store instruction handles both row and column tails
        - Reduces branches and special-case code in tail processing
        - block-2D also supports descriptor-based remainder handling

    30. Descriptor-based remainder
        - Handle boundaries by changing block_2d width/height descriptor fields
        - Zero extra ALU overhead (compared with mask-register predication)
        - Applicable only to block-2D access types

    [Register Allocation]

    31. Bank-aware GRF allocation
        - Avoid placing A/B operands and accumulators in the same GRF bank (bank conflict)
        - Practical rule: place A in even bank, B in odd bank, C accumulators flexible
        - In OpenCL: use register-allocation attributes (if available) or manual register-pressure shaping

    32. Choosing GRF count: 128 vs 256
        - 256 GRF → larger tiles and higher compute density, but occupancy drops to ~4 threads/EU
        - 128 GRF → smaller tiles, but ~8 threads/EU can hide latency better
        - For bandwidth-bound cases (small K), prefer 128; for compute-bound cases (large K), prefer 256

    [Misc]

    33. TLB warmup for large matrices
        - Issue one dummy load per A/B page before entering k-loop
        - Prefills TLB and avoids page-walk stalls in main loop
        - Usually helps once A/B footprint exceeds ~256KB

    34. Boustrophedon internal FMA ordering (fmaBoustrophedon)
        - Arrange DPAS/FMA order in snake pattern: one pass forward, next pass reverse
        - Improves register locality (neighboring DPAS reuses source registers)
        - Reduces RAW stalls

    35. cLoadAhead (C preload)
        - If beta ≠ 0, preload current C tile in advance
        - Overlap C-load latency with tail part of k-loop DPAS
        - Then directly compute fma(alpha, acc, beta, C_old) and write back

    36. Avoid unnecessary repack
        - If A/B are already in DPAS-friendly memory layout (packed, crosspack=2):
        load directly to GRF via block-load without extra SLM copy + repack
        - Quick check: isPacked(A.layout) && alignment >= 16

    37. Kernel catalog / auto-tuning mindset
        - Use different tile/strategy combinations for different (M,N,K) ranges
        - Small M (decode): smaller unrollM, larger unrollN, possibly no-SLM, Stream-K
        - Large M (prefill): larger unrollM, SLM double buffer, Boustrophedon walk

- Provide explicit launch metadata (GWS/LWS/subgroup hints) in kernel comments.
- Please update task.py to align with kernel's GWS/LWS and other assumptions if you make significant changes to the kernel structure.

## Reference code / Task:

This is the reference OCL implementation:
```
// Simple row-major FP16 matmul:
//   C[M,N] = A[M,K] x B[K,N]
// Input/Output dtype: half
// Accumulation dtype: float
__kernel void matmul(
    __global const half* A,
    __global const half* B,
    __global half* C,
    const int M,
    const int K,
    const int N)
{
    const int col = get_global_id(0);
    const int row = get_global_id(1);

    if (row >= M || col >= N)
        return;

    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        acc += convert_float(A[row * K + k]) * convert_float(B[k * N + col]);
    }

    C[row * N + col] = convert_half(acc);
}

```

## Previous OCL implementations with scores:

### Version 1 (Result: Correctness score: 5 / 5 (compiles and correct), runtime in ms: 1.450):
```OCL
// GEMM: C[M,N] = A[M,K] x B[K,N], all half precision, f32 accumulation
// Architecture: 64 WIs (4 SGs of 16), TILE 32x64x32, A in SLM double-buffered, B from global
// Launch: LWS=(16,4,1), GWS=(N/4, M/8, 1) i.e. (N/64*16, M/32*4, 1)
// Subgroup size: 16
// For M=N=K=4096: GWS=(1024,512,1), LWS=(16,4,1)

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void matmul(
    __global const half* restrict A,
    __global const half* restrict B,
    __global half* restrict C,
    const int M,
    const int K,
    const int N)
{
    // Work-group tile: 32 rows x 64 cols
    // 4 subgroups, each handles 32 rows x 16 cols
    const int sg_id = get_local_id(1);       // 0..3 (which subgroup)
    const int sg_lid = get_sub_group_local_id(); // 0..15 (lane within SG)

    // WG position in output
    const int wg_row = get_group_id(1) * 32;  // M tile start
    const int wg_col = get_group_id(0) * 64;  // N tile start (get_group_id(0) = global_id(0)/16)

    // Each subgroup handles 16 consecutive columns
    const int sg_col = wg_col + sg_id * 16;

    // SLM for A tile: double-buffered, 32 rows x 32 cols (half)
    // Stride = 32 halfs (64 bytes) - allows block reads
    #define SLM_STRIDE 32
    __local half slm_A[2 * 32 * SLM_STRIDE];  // 2 * 32 * 32 * 2 bytes = 4096 bytes

    // Accumulators: 32 rows x 16 cols per subgroup = 4 DPAS results of float8
    float8 acc0 = 0.0f;  // rows 0-7
    float8 acc1 = 0.0f;  // rows 8-15
    float8 acc2 = 0.0f;  // rows 16-23
    float8 acc3 = 0.0f;  // rows 24-31

    // Linear local ID for cooperative loading
    const int lid = sg_id * 16 + sg_lid;  // 0..63

    // A load mapping: 64 WIs load 32x32 = 1024 halfs, 16 per WI
    // lid/2 = row (0..31), (lid%2)*16 = col offset (0 or 16)
    const int a_load_row = lid / 2;
    const int a_load_col_off = (lid % 2) * 16;

    // Preload first A tile into SLM buffer 0
    {
        int a_global_offset = (wg_row + a_load_row) * K + a_load_col_off;
        __global const half* a_ptr = A + a_global_offset;

        half8 v0 = vload8(0, a_ptr);
        half8 v1 = vload8(1, a_ptr);

        int slm_offset = a_load_row * SLM_STRIDE + a_load_col_off;
        vstore8(v0, 0, &slm_A[slm_offset]);
        vstore8(v1, 0, &slm_A[slm_offset + 8]);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    int buf = 0;  // current buffer being consumed
    const int k_iterations = K / 32;

    for (int ki = 0; ki < k_iterations; ki++) {
        int k = ki * 32;
        int next_k = k + 32;
        int next_buf = 1 - buf;

        // === COMPUTE from current SLM buffer ===
        // Process two K16 steps from the 32-wide K tile

        __global const half* b_base = B + k * N + sg_col + sg_lid;

        // --- K16 step 0 (k=0..15 within tile) ---
        int8 b_reg0;
        {
            // Load B[k:k+16, sg_col+sg_lid] and pack into VNNI int8
            // Each pair of consecutive k-rows packs into one int
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                half v0 = b_base[(2*i) * N];
                half v1 = b_base[(2*i+1) * N];
                ((int*)&b_reg0)[i] = as_int((short2)(as_short(v0), as_short(v1)));
            }
        }

        // Load A from SLM for K16 step 0
        int slm_base = buf * 32 * SLM_STRIDE;
        short8 a0_0, a1_0, a2_0, a3_0;

        // rows 0-7, k=0..15: lane sg_lid reads k=sg_lid
        {
            __local const half* a_slm = &slm_A[slm_base + sg_lid];
            a0_0.s0 = as_short(a_slm[0*SLM_STRIDE]);
            a0_0.s1 = as_short(a_slm[1*SLM_STRIDE]);
            a0_0.s2 = as_short(a_slm[2*SLM_STRIDE]);
            a0_0.s3 = as_short(a_slm[3*SLM_STRIDE]);
            a0_0.s4 = as_short(a_slm[4*SLM_STRIDE]);
            a0_0.s5 = as_short(a_slm[5*SLM_STRIDE]);
            a0_0.s6 = as_short(a_slm[6*SLM_STRIDE]);
            a0_0.s7 = as_short(a_slm[7*SLM_STRIDE]);
        }
        {
            __local const half* a_slm = &slm_A[slm_base + 8*SLM_STRIDE + sg_lid];
            a1_0.s0 = as_short(a_slm[0*SLM_STRIDE]);
            a1_0.s1 = as_short(a_slm[1*SLM_STRIDE]);
            a1_0.s2 = as_short(a_slm[2*SLM_STRIDE]);
            a1_0.s3 = as_short(a_slm[3*SLM_STRIDE]);
            a1_0.s4 = as_short(a_slm[4*SLM_STRIDE]);
            a1_0.s5 = as_short(a_slm[5*SLM_STRIDE]);
            a1_0.s6 = as_short(a_slm[6*SLM_STRIDE]);
            a1_0.s7 = as_short(a_slm[7*SLM_STRIDE]);
        }
        {
            __local const half* a_slm = &slm_A[slm_base + 16*SLM_STRIDE + sg_lid];
            a2_0.s0 = as_short(a_slm[0*SLM_STRIDE]);
            a2_0.s1 = as_short(a_slm[1*SLM_STRIDE]);
            a2_0.s2 = as_short(a_slm[2*SLM_STRIDE]);
            a2_0.s3 = as_short(a_slm[3*SLM_STRIDE]);
            a2_0.s4 = as_short(a_slm[4*SLM_STRIDE]);
            a2_0.s5 = as_short(a_slm[5*SLM_STRIDE]);
            a2_0.s6 = as_short(a_slm[6*SLM_STRIDE]);
            a2_0.s7 = as_short(a_slm[7*SLM_STRIDE]);
        }
        {
            __local const half* a_slm = &slm_A[slm_base + 24*SLM_STRIDE + sg_lid];
            a3_0.s0 = as_short(a_slm[0*SLM_STRIDE]);
            a3_0.s1 = as_short(a_slm[1*SLM_STRIDE]);
            a3_0.s2 = as_short(a_slm[2*SLM_STRIDE]);
            a3_0.s3 = as_short(a_slm[3*SLM_STRIDE]);
            a3_0.s4 = as_short(a_slm[4*SLM_STRIDE]);
            a3_0.s5 = as_short(a_slm[5*SLM_STRIDE]);
            a3_0.s6 = as_short(a_slm[6*SLM_STRIDE]);
            a3_0.s7 = as_short(a_slm[7*SLM_STRIDE]);
        }

        // DPAS K16 step 0
        acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0_0, b_reg0, acc0);
        acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1_0, b_reg0, acc1);
        acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2_0, b_reg0, acc2);
        acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3_0, b_reg0, acc3);

        // --- K16 step 1 (k=16..31 within tile) ---
        __global const half* b_base2 = b_base + 16 * N;
        int8 b_reg1;
        {
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                half v0 = b_base2[(2*i) * N];
                half v1 = b_base2[(2*i+1) * N];
                ((int*)&b_reg1)[i] = as_int((short2)(as_short(v0), as_short(v1)));
            }
        }

        // Load A from SLM for K16 step 1 (offset by 16 in k)
        short8 a0_1, a1_1, a2_1, a3_1;
        {
            __local const half* a_slm = &slm_A[slm_base + 16 + sg_lid];
            a0_1.s0 = as_short(a_slm[0*SLM_STRIDE]);
            a0_1.s1 = as_short(a_slm[1*SLM_STRIDE]);
            a0_1.s2 = as_short(a_slm[2*SLM_STRIDE]);
            a0_1.s3 = as_short(a_slm[3*SLM_STRIDE]);
            a0_1.s4 = as_short(a_slm[4*SLM_STRIDE]);
            a0_1.s5 = as_short(a_slm[5*SLM_STRIDE]);
            a0_1.s6 = as_short(a_slm[6*SLM_STRIDE]);
            a0_1.s7 = as_short(a_slm[7*SLM_STRIDE]);
        }
        {
            __local const half* a_slm = &slm_A[slm_base + 8*SLM_STRIDE + 16 + sg_lid];
            a1_1.s0 = as_short(a_slm[0*SLM_STRIDE]);
            a1_1.s1 = as_short(a_slm[1*SLM_STRIDE]);
            a1_1.s2 = as_short(a_slm[2*SLM_STRIDE]);
            a1_1.s3 = as_short(a_slm[3*SLM_STRIDE]);
            a1_1.s4 = as_short(a_slm[4*SLM_STRIDE]);
            a1_1.s5 = as_short(a_slm[5*SLM_STRIDE]);
            a1_1.s6 = as_short(a_slm[6*SLM_STRIDE]);
            a1_1.s7 = as_short(a_slm[7*SLM_STRIDE]);
        }
        {
            __local const half* a_slm = &slm_A[slm_base + 16*SLM_STRIDE + 16 + sg_lid];
            a2_1.s0 = as_short(a_slm[0*SLM_STRIDE]);
            a2_1.s1 = as_short(a_slm[1*SLM_STRIDE]);
            a2_1.s2 = as_short(a_slm[2*SLM_STRIDE]);
            a2_1.s3 = as_short(a_slm[3*SLM_STRIDE]);
            a2_1.s4 = as_short(a_slm[4*SLM_STRIDE]);
            a2_1.s5 = as_short(a_slm[5*SLM_STRIDE]);
            a2_1.s6 = as_short(a_slm[6*SLM_STRIDE]);
            a2_1.s7 = as_short(a_slm[7*SLM_STRIDE]);
        }
        {
            __local const half* a_slm = &slm_A[slm_base + 24*SLM_STRIDE + 16 + sg_lid];
            a3_1.s0 = as_short(a_slm[0*SLM_STRIDE]);
            a3_1.s1 = as_short(a_slm[1*SLM_STRIDE]);
            a3_1.s2 = as_short(a_slm[2*SLM_STRIDE]);
            a3_1.s3 = as_short(a_slm[3*SLM_STRIDE]);
            a3_1.s4 = as_short(a_slm[4*SLM_STRIDE]);
            a3_1.s5 = as_short(a_slm[5*SLM_STRIDE]);
            a3_1.s6 = as_short(a_slm[6*SLM_STRIDE]);
            a3_1.s7 = as_short(a_slm[7*SLM_STRIDE]);
        }

        // DPAS K16 step 1
        acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0_1, b_reg1, acc0);
        acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1_1, b_reg1, acc1);
        acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2_1, b_reg1, acc2);
        acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3_1, b_reg1, acc3);

        // === Load next A tile into other SLM buffer ===
        if (next_k < K) {
            int a_global_offset = (wg_row + a_load_row) * K + next_k + a_load_col_off;
            __global const half* a_ptr = A + a_global_offset;
            half8 v0 = vload8(0, a_ptr);
            half8 v1 = vload8(1, a_ptr);

            int slm_offset = next_buf * 32 * SLM_STRIDE + a_load_row * SLM_STRIDE + a_load_col_off;
            vstore8(v0, 0, &slm_A[slm_offset]);
            vstore8(v1, 0, &slm_A[slm_offset + 8]);

            buf = next_buf;
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // === Store C ===
    // Each subgroup writes 32 rows x 16 cols
    // acc[r] at lane sg_lid = C[wg_row + row_offset + r, sg_col + sg_lid]
    __global half* c_ptr = C + wg_row * N + sg_col + sg_lid;

    c_ptr[0*N] = convert_half(acc0.s0);
    c_ptr[1*N] = convert_half(acc0.s1);
    c_ptr[2*N] = convert_half(acc0.s2);
    c_ptr[3*N] = convert_half(acc0.s3);
    c_ptr[4*N] = convert_half(acc0.s4);
    c_ptr[5*N] = convert_half(acc0.s5);
    c_ptr[6*N] = convert_half(acc0.s6);
    c_ptr[7*N] = convert_half(acc0.s7);

    c_ptr[8*N] = convert_half(acc1.s0);
    c_ptr[9*N] = convert_half(acc1.s1);
    c_ptr[10*N] = convert_half(acc1.s2);
    c_ptr[11*N] = convert_half(acc1.s3);
    c_ptr[12*N] = convert_half(acc1.s4);
    c_ptr[13*N] = convert_half(acc1.s5);
    c_ptr[14*N] = convert_half(acc1.s6);
    c_ptr[15*N] = convert_half(acc1.s7);

    c_ptr[16*N] = convert_half(acc2.s0);
    c_ptr[17*N] = convert_half(acc2.s1);
    c_ptr[18*N] = convert_half(acc2.s2);
    c_ptr[19*N] = convert_half(acc2.s3);
    c_ptr[20*N] = convert_half(acc2.s4);
    c_ptr[21*N] = convert_half(acc2.s5);
    c_ptr[22*N] = convert_half(acc2.s6);
    c_ptr[23*N] = convert_half(acc2.s7);

    c_ptr[24*N] = convert_half(acc3.s0);
    c_ptr[25*N] = convert_half(acc3.s1);
    c_ptr[26*N] = convert_half(acc3.s2);
    c_ptr[27*N] = convert_half(acc3.s3);
    c_ptr[28*N] = convert_half(acc3.s4);
    c_ptr[29*N] = convert_half(acc3.s5);
    c_ptr[30*N] = convert_half(acc3.s6);
    c_ptr[31*N] = convert_half(acc3.s7);
}
```

## Best implementation so far:

This is the code of the best kernel we have tested so far (Result: Correctness score: 5 / 5 (compiles and correct), runtime in ms: 1.120):
```OCL
// GEMM: C[M,N] = A[M,K] x B[K,N], all half precision, f32 accumulation
// Architecture: 64 WIs (4 SGs of 16), TILE 32x64x32, A in SLM double-buffered, B from global
// Launch: LWS=(16,4,1), GWS=(N/4, M/8, 1) i.e. (N/64*16, M/32*4, 1)
// Subgroup size: 16
// For M=N=K=4096: GWS=(1024,512,1), LWS=(16,4,1)

#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void matmul(
    __global const half* restrict A,
    __global const half* restrict B,
    __global half* restrict C,
    const int M,
    const int K,
    const int N)
{
    // Work-group tile: 32 rows x 64 cols
    // 4 subgroups, each handles 32 rows x 16 cols
    const int sg_id = get_local_id(1);       // 0..3 (which subgroup)
    const int sg_lid = get_sub_group_local_id(); // 0..15 (lane within SG)

    // WG position in output
    const int wg_row = get_group_id(1) * 32;  // M tile start
    const int wg_col = get_group_id(0) * 64;  // N tile start (get_group_id(0) = global_id(0)/16 effectively)

    // Each subgroup handles 16 consecutive columns
    const int sg_col = wg_col + sg_id * 16;

    // SLM for A tile: double-buffered
    // Layout: two K16 blocks per buffer, each block is 32 rows x 16 cols with stride=16
    // This enables efficient subgroup reads
    // Buffer layout: [buf][k_half][row][k_within_16]
    // Total per buffer: 32 * 32 = 1024 halfs
    // With stride=16 for each k_half block: 2 * 32 * 16 = 1024 halfs per buffer
    // Double buffer: 2048 halfs total = 4096 bytes
    #define SLM_K16_STRIDE 16
    #define SLM_BUF_SIZE 1024
    __local half slm_A[2 * SLM_BUF_SIZE];  // double buffer, each 32*32 halfs

    // Accumulators: 32 rows x 16 cols per subgroup = 4 DPAS results of float8
    float8 acc0 = 0.0f;  // rows 0-7
    float8 acc1 = 0.0f;  // rows 8-15
    float8 acc2 = 0.0f;  // rows 16-23
    float8 acc3 = 0.0f;  // rows 24-31

    // Linear local ID for cooperative loading
    const int lid = sg_id * 16 + sg_lid;  // 0..63

    // A load mapping: 64 WIs load 32x32 = 1024 halfs = 16 per WI
    // Map: lid/2 = row (0..31), (lid%2) = which k_half (0 or 1)
    // Each WI loads 16 consecutive halfs (one row of one k_half block)
    const int a_load_row = lid / 2;        // 0..31
    const int a_load_khalf = lid % 2;      // 0 or 1

    // Preload first A tile into SLM buffer 0
    {
        int global_row = wg_row + a_load_row;
        int global_k = a_load_khalf * 16;
        __global const half* a_ptr = A + global_row * K + global_k;

        // SLM offset: k_half * 32*16 + row * 16
        int slm_off = a_load_khalf * (32 * SLM_K16_STRIDE) + a_load_row * SLM_K16_STRIDE;

        // Load 16 halfs using vload8 x2
        half8 v0 = vload8(0, a_ptr);
        half8 v1 = vload8(1, a_ptr);
        vstore8(v0, 0, &slm_A[slm_off]);
        vstore8(v1, 0, &slm_A[slm_off + 8]);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    int buf = 0;  // current buffer being consumed
    const int k_iterations = K / 32;

    for (int ki = 0; ki < k_iterations; ki++) {
        int k_offset = ki * 32;
        int next_k = k_offset + 32;

        // === COMPUTE from current SLM buffer ===
        int slm_base = buf * SLM_BUF_SIZE;

        // B pointer for this K-step
        __global const half* b_base = B + k_offset * N + sg_col + sg_lid;

        // --- K16 step 0 (k=0..15) ---
        {
            // Load A from SLM: k_half=0 block
            // A[row][k] at slm_A[slm_base + 0*(32*16) + row*16 + k]
            // For subgroup block read: base at slm_base + row_base*16
            // Lane sg_lid reads k=sg_lid, short8 gives 8 rows
            int a_off = slm_base;

            short8 a0, a1, a2, a3;

            // Use intel_sub_group_block_read_us to read 8 consecutive ushorts
            // from SLM with stride=16 between vector elements
            // This reads: lane i gets element at base + i, base+16+i, ..., base+7*16+i
            // Which gives us A[row_base+j][k=sg_lid] for j=0..7

            // rows 0-7
            a0 = as_short8(intel_sub_group_block_read_us8((__local const uint*)&slm_A[a_off + 0*SLM_K16_STRIDE]));
            // rows 8-15
            a1 = as_short8(intel_sub_group_block_read_us8((__local const uint*)&slm_A[a_off + 8*SLM_K16_STRIDE]));
            // rows 16-23
            a2 = as_short8(intel_sub_group_block_read_us8((__local const uint*)&slm_A[a_off + 16*SLM_K16_STRIDE]));
            // rows 24-31
            a3 = as_short8(intel_sub_group_block_read_us8((__local const uint*)&slm_A[a_off + 24*SLM_K16_STRIDE]));

            // Load B[k:k+16, sg_col+sg_lid] and pack into VNNI int8
            // Each lane loads 16 values from its column, stride=N between rows
            // Pack pairs into int: b[i] = pack(B[2i], B[2i+1])
            int8 b_reg;

            __global const half* bp = b_base;
            b_reg.s0 = as_int((half2)(bp[0], bp[N])); bp += 2*N;
            b_reg.s1 = as_int((half2)(bp[0], bp[N])); bp += 2*N;
            b_reg.s2 = as_int((half2)(bp[0], bp[N])); bp += 2*N;
            b_reg.s3 = as_int((half2)(bp[0], bp[N])); bp += 2*N;
            b_reg.s4 = as_int((half2)(bp[0], bp[N])); bp += 2*N;
            b_reg.s5 = as_int((half2)(bp[0], bp[N])); bp += 2*N;
            b_reg.s6 = as_int((half2)(bp[0], bp[N])); bp += 2*N;
            b_reg.s7 = as_int((half2)(bp[0], bp[N]));

            // DPAS calls
            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b_reg, acc0);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b_reg, acc1);
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b_reg, acc2);
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b_reg, acc3);
        }

        // --- K16 step 1 (k=16..31) ---
        {
            // Load A from SLM: k_half=1 block
            int a_off = slm_base + 32 * SLM_K16_STRIDE;

            short8 a0, a1, a2, a3;

            a0 = as_short8(intel_sub_group_block_read_us8((__local const uint*)&slm_A[a_off + 0*SLM_K16_STRIDE]));
            a1 = as_short8(intel_sub_group_block_read_us8((__local const uint*)&slm_A[a_off + 8*SLM_K16_STRIDE]));
            a2 = as_short8(intel_sub_group_block_read_us8((__local const uint*)&slm_A[a_off + 16*SLM_K16_STRIDE]));
            a3 = as_short8(intel_sub_group_block_read_us8((__local const uint*)&slm_A[a_off + 24*SLM_K16_STRIDE]));

            // Load B for k=16..31
            __global const half* bp = b_base + 16 * N;
            int8 b_reg;

            b_reg.s0 = as_int((half2)(bp[0], bp[N])); bp += 2*N;
            b_reg.s1 = as_int((half2)(bp[0], bp[N])); bp += 2*N;
            b_reg.s2 = as_int((half2)(bp[0], bp[N])); bp += 2*N;
            b_reg.s3 = as_int((half2)(bp[0], bp[N])); bp += 2*N;
            b_reg.s4 = as_int((half2)(bp[0], bp[N])); bp += 2*N;
            b_reg.s5 = as_int((half2)(bp[0], bp[N])); bp += 2*N;
            b_reg.s6 = as_int((half2)(bp[0], bp[N])); bp += 2*N;
            b_reg.s7 = as_int((half2)(bp[0], bp[N]));

            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b_reg, acc0);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b_reg, acc1);
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b_reg, acc2);
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b_reg, acc3);
        }

        // === Load next A tile into other SLM buffer ===
        if (next_k < K) {
            int next_buf = 1 - buf;
            int global_row = wg_row + a_load_row;
            int global_k = next_k + a_load_khalf * 16;
            __global const half* a_ptr = A + global_row * K + global_k;

            int slm_off = next_buf * SLM_BUF_SIZE + a_load_khalf * (32 * SLM_K16_STRIDE) + a_load_row * SLM_K16_STRIDE;

            half8 v0 = vload8(0, a_ptr);
            half8 v1 = vload8(1, a_ptr);
            vstore8(v0, 0, &slm_A[slm_off]);
            vstore8(v1, 0, &slm_A[slm_off + 8]);

            buf = next_buf;
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // === Store C ===
    // Each subgroup writes 32 rows x 16 cols
    // acc[r] at lane sg_lid = C[wg_row + row_offset + r, sg_col + sg_lid]
    __global half* c_ptr = C + wg_row * N + sg_col + sg_lid;

    // Store rows 0-7
    c_ptr[0*N] = convert_half(acc0.s0);
    c_ptr[1*N] = convert_half(acc0.s1);
    c_ptr[2*N] = convert_half(acc0.s2);
    c_ptr[3*N] = convert_half(acc0.s3);
    c_ptr[4*N] = convert_half(acc0.s4);
    c_ptr[5*N] = convert_half(acc0.s5);
    c_ptr[6*N] = convert_half(acc0.s6);
    c_ptr[7*N] = convert_half(acc0.s7);

    // Store rows 8-15
    c_ptr[8*N] = convert_half(acc1.s0);
    c_ptr[9*N] = convert_half(acc1.s1);
    c_ptr[10*N] = convert_half(acc1.s2);
    c_ptr[11*N] = convert_half(acc1.s3);
    c_ptr[12*N] = convert_half(acc1.s4);
    c_ptr[13*N] = convert_half(acc1.s5);
    c_ptr[14*N] = convert_half(acc1.s6);
    c_ptr[15*N] = convert_half(acc1.s7);

    // Store rows 16-23
    c_ptr[16*N] = convert_half(acc2.s0);
    c_ptr[17*N] = convert_half(acc2.s1);
    c_ptr[18*N] = convert_half(acc2.s2);
    c_ptr[19*N] = convert_half(acc2.s3);
    c_ptr[20*N] = convert_half(acc2.s4);
    c_ptr[21*N] = convert_half(acc2.s5);
    c_ptr[22*N] = convert_half(acc2.s6);
    c_ptr[23*N] = convert_half(acc2.s7);

    // Store rows 24-31
    c_ptr[24*N] = convert_half(acc3.s0);
    c_ptr[25*N] = convert_half(acc3.s1);
    c_ptr[26*N] = convert_half(acc3.s2);
    c_ptr[27*N] = convert_half(acc3.s3);
    c_ptr[28*N] = convert_half(acc3.s4);
    c_ptr[29*N] = convert_half(acc3.s5);
    c_ptr[30*N] = convert_half(acc3.s6);
    c_ptr[31*N] = convert_half(acc3.s7);
}
```

### Previous kernel and evaluation:
Here is the last OCL kernel we tested (Result: Correctness score: 3 / 5 (compiled but incorrect), runtime in ms: -1.000):
```OCL
// GEMM: C[M,N] = A[M,K] x B[K,N], all half precision, f32 accumulation
// Architecture: 64 WIs (4 SGs of 16), TILE 32x64x32, A in SLM, B from global
// Launch: GWS = (N/64 * 64, M/32), LWS = (64, 1)
// Subgroup size: 16

#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void matmul(
    __global const half* A,
    __global const half* B,
    __global half* C,
    const int M,
    const int K,
    const int N)
{
    // Work-group tile position
    const int wg_n = (get_group_id(0)) * 64;  // N tile start
    const int wg_m = get_group_id(1) * 32;     // M tile start

    const int lid = get_local_id(0);           // 0..63
    const int sg_id = get_sub_group_id();      // 0..3
    const int sg_lid = get_sub_group_local_id(); // 0..15

    // SLM for A tile: 32 rows x 32 K-elements (with padding to avoid bank conflicts)
    // Stride = 34 (32 + 2 padding) to avoid bank conflicts
    #define SLM_A_STRIDE 34
    __local half slm_A[32 * SLM_A_STRIDE];  // Double buffer not needed if we barrier properly

    // Each SG handles 16 columns of N (sg_id determines which 16 columns)
    // Each SG computes 32 rows x 16 cols of C
    // That requires 4 DPAS calls in M direction (each produces 8 rows)
    // Accumulators: 4 x float8
    float8 acc0 = 0.0f;  // rows 0-7
    float8 acc1 = 0.0f;  // rows 8-15
    float8 acc2 = 0.0f;  // rows 16-23
    float8 acc3 = 0.0f;  // rows 24-31

    // B base for this SG's 16 columns
    const int b_col = wg_n + sg_id * 16;

    // K-loop: step by 32
    for (int k = 0; k < K; k += 32) {
        // === Cooperative load A[wg_m:wg_m+32, k:k+32] into SLM ===
        // 64 WIs load 32*32 = 1024 half values = 16 per WI
        // Each WI loads 16 consecutive halfs
        // Layout: row-major in SLM with stride SLM_A_STRIDE
        // 1024 elements / 64 WIs = 16 elements per WI
        // Strategy: each WI handles specific rows/cols
        // With 64 WIs and 32x32 tile: 
        //   lid / 2 = row (0..31), (lid % 2) * 16 = col offset (0 or 16)
        {
            int a_row = lid / 2;          // 0..31
            int a_col_base = (lid % 2) * 16;  // 0 or 16

            int global_row = wg_m + a_row;
            int global_col = k + a_col_base;

            __global const half* a_ptr = A + global_row * K + global_col;

            // Load 16 halfs (32 bytes) - use vload8 x2 for efficiency
            half8 v0 = vload8(0, a_ptr);
            half8 v1 = vload8(1, a_ptr);

            // Store to SLM
            __local half* slm_ptr = slm_A + a_row * SLM_A_STRIDE + a_col_base;
            vstore8(v0, 0, slm_ptr);
            vstore8(v1, 1, slm_ptr);
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // === Load B and compute DPAS ===
        // B tile: 32 rows x 16 cols (for this SG)
        // DPAS needs B in packed format: int8 means 8 ints = 8*2 halfs = 16 halfs per lane
        // For k16: B is [16, 16] packed as int8 per subgroup lane
        // intel_sub_group_f16_f16_matrix_mad_k16(short8 a, int8 b, float8 acc)
        //   a: short8 = 8 rows x k16 (packed as 8 shorts, each short = 1 half... no)
        //   Actually: short8 a = 8 pairs of f16 for the A operand (8 rows, k16 means 16 k-elements)
        //   Wait - let me reconsider the DPAS intrinsic signature

        // For Xe2 with exec_size=16:
        // intel_sub_group_f16_f16_matrix_mad_k16(short8 a, int8 b, float8 acc)
        // - a (short8): A matrix fragment, 8 rows × 16 K elements packed
        //   Each short = 2 f16 values (crosspack=2), so short8 = 8 × 2 = 16 f16 per WI
        //   Across 16 WIs... actually a is replicated/shared
        //   A is [8 rows × 16 K], packed: each row has 16 f16 = 8 shorts
        //   short8 per WI represents one row? No...
        //   
        // Let me think about this more carefully:
        // DPAS computes C[8,16] += A[8,16] * B[16,16]
        // - C: float8 per WI, 16 WIs = 8×16 output
        // - A: short8 per WI, but A is shared across the subgroup (broadcast)
        //   short8 = 8 values, each short holds 2 f16 (crosspack=2)
        //   So short8 = 8×2 = 16 f16 values... that's one row of A (16 K elements)
        //   But we need 8 rows... 
        //   Actually short8 = 8 shorts = 8 rows, each short = 2 f16 (k-pair)
        //   With k16, we need 8 such calls? No...
        //   
        // The correct interpretation for intel_sub_group_f16_f16_matrix_mad_k16:
        // - Performs C[8,exec_size] += A[8,16] * B[16,exec_size]
        // - a (short8): A matrix, 8 rows × 16 cols, packed as:
        //   Each WI holds short8 = 8 shorts. Since k=16 and crosspack=2,
        //   we need 16/2 = 8 pairs per row... but that's 8 shorts per row and we have 8 rows.
        //   So total = 8*8 = 64 shorts... that doesn't fit in short8.
        //   
        // Actually I think the A operand is distributed differently:
        // - short8 per WI: the 16 WIs together hold the full A matrix
        //   Each WI holds 8 shorts = 8 rows × 1 k-pair (2 f16 values)
        //   16 WIs × 2 f16 = 32 f16 per row... that's too many for k=16
        //   
        // Let me look at this from the oneDNN perspective:
        // For Xe2 (exec_size=16), repcount=8, sdepth=8:
        // - One DPAS instruction: C[8,16] += A[8,16] * B[16,16]  (k=16 = sdepth*2 for f16)
        // - A operand: 8 rows × 16 k-elements
        //   Packed as: short8 per WI, where each WI contributes to different k-columns
        //   With 16 WIs and k=16: each WI holds k-index = wi_id
        //   short8 = 8 rows, each short = 2 f16 (the pair at k=2*floor(wi_id/1)... )
        //   
        // I think the simplest correct interpretation is:
        // A[8,16] stored in short8: 
        //   - 8 shorts = 8 rows
        //   - Each short = 2 consecutive f16 in k-dimension (crosspack=2)
        //   - The subgroup lane selects which k-pair: lane i selects k-pair i (but 16 lanes × 2 = 32 > 16)
        //   
        // Hmm, let me try: for k16 with exec_size=16:
        //   - sdepth = 8 (systolic depth)
        //   - ops_per_chan = 2 (for f16: 2 f16 per 32-bit channel)
        //   - k = sdepth * ops_per_chan = 16
        //   - A is "src1": repcount × sdepth × ops_per_chan = 8 × 8 × 2 = 128 f16 values total
        //     But distributed across... 
        //   
        // For the A operand in DPAS (src1):
        //   - Size per WI: repcount * sdepth * ops_per_chan / exec_size... no
        //   - Actually A is NOT distributed across lanes in the same way as B
        //   - A is replicated: each lane gets the same A data
        //   - So short8 = 8 rows × (2 f16 per short) = 8 rows × 1 k-pair
        //   - But k=16 means 8 k-pairs... so we'd need short8 × 8 calls?
        //   
        // I think I'm overcomplicating this. Let me use the practical approach:
        // 
        // From Intel documentation and oneDNN usage:
        // intel_sub_group_f16_f16_matrix_mad_k16(short8 a, int8 b, float8 acc)
        // - Computes: acc += A * B where result is 8 rows × 16 cols (exec_size=16)
        // - a (short8): A matrix data for this WI
        //   8 shorts per WI × 16 WIs = 128 shorts = 256 bytes
        //   A is [8 rows × 16 K]: 8*16 = 128 f16 = 128 shorts = 256 bytes ✓
        //   So each WI holds 8 shorts = 8 f16 values from A
        //   Distribution: WI i holds A[row][k] where... 
        //   With crosspack=2: pairs of k are in same 32-bit word
        //   short8 per WI: element j (0..7) = A[j][2*wi_id : 2*wi_id+1] packed as short
        //   Wait, that gives 16 WIs × 2 = 32 k-elements per row, but we only have 16.
        //   
        //   Alternative: short8 per WI: element j = A[j][wi_id] as a half stored in short
        //   16 WIs × 1 = 16 k-elements per row ✓, 8 elements = 8 rows ✓
        //   But then it's not "crosspack=2"...
        //   
        // OK let me just look at how to load A from SLM for DPAS:
        // A is stored row-major in SLM: A[row][k], row=0..7, k=0..15
        // For DPAS, A needs to be in "VNNI" / crosspack=2 format:
        //   Pairs of k-elements are packed: A[row][2k] and A[row][2k+1] in one 32-bit word
        //   Layout: [row][k/2] where each element is {A[row][2*(k/2)], A[row][2*(k/2)+1]}
        //   
        // For short8 a: each WI holds 8 shorts
        //   WI i holds: a[j] = as_short((half2)(A[j][2*i], A[j][2*i+1])) ... no, short can only hold 1 half
        //   
        // Actually, I think for the A operand:
        //   short8 means 8 elements of type short (16-bit)
        //   Each short IS one f16 value (reinterpreted)
        //   The distribution: WI i holds A[row 0..7][k = i] ... but that's only k=0..15 for 16 WIs
        //   Hmm but k=16 and we have 16 WIs, so each WI holds one k-column of A
        //   short8 = 8 rows of that k-column
        //   
        // Wait, but then how does crosspack=2 work? Let me try another interpretation:
        //   short8 per WI: 8 shorts = 16 bytes
        //   Total across SG: 16 WIs × 16 bytes = 256 bytes = 128 f16 = 8×16 ✓
        //   
        //   If A is stored in SLM with crosspack=2 (pairs of k packed):
        //   Layout: for each row r (0..7), for each k-pair p (0..7):
        //     address = r * 16 + p * 2  (in f16 units)
        //     stores: A[r][2p], A[r][2p+1]
        //   
        //   Then intel_sub_group_block_read from SLM:
        //   WI i reads consecutive 16-bit elements at offset i*2 bytes from base
        //   If we read short8, WI i gets 8 shorts at addresses base + i*2, base + 32 + i*2, ...
        //   (stride = 32 bytes = 16 shorts between consecutive elements of the vector)
        //   
        //   Hmm, this is getting complex. Let me just use a simpler approach:
        //   Load A from SLM into short8 using intel_sub_group_block_read_us8
        //   which reads 8 consecutive ushort values per WI with subgroup stride

        // Let me use a simpler, proven approach:
        // Load A from SLM: for each of 8 rows, read 16 halfs (one per WI via subgroup read)
        // This gives short8 a where a[i] = A[row_base + i][k_base + sg_lid]
        // But this only covers k=0..15 (one k-step of 16)

        // For B: B[16,16] for this subgroup
        // int8 b: 8 ints per WI × 16 WIs = 128 ints = 512 bytes = 256 f16
        // B is [16 K-rows × 16 N-cols]: 16*16 = 256 f16 ✓
        // With VNNI/crosspack=2: B[k/2][n] where each element packs B[2*(k/2)][n] and B[2*(k/2)+1][n]
        // int (32-bit) holds 2 f16 values
        // int8 per WI: WI i holds B[k-pairs 0..7][col i] = 8 k-pairs for column i
        // 16 WIs × 1 col = 16 cols ✓, 8 k-pairs × 2 = 16 k-rows ✓

        // So for B in VNNI format:
        // B_vnni[k/2][n] = pack(B[2*(k/2)][n], B[2*(k/2)+1][n]) as int
        // Load: intel_sub_group_block_read8 from B_vnni gives int8 per WI

        // But our B is in row-major f16 format (not pre-packed VNNI)!
        // We need to load B and repack on the fly, or load pairs of rows.

        // For B row-major: B[k][n], stride = N
        // To get VNNI format for DPAS:
        // We need to load B[k][col..col+15] for k=0..15 and pack pairs
        // 
        // Load approach for B (no SLM, direct from global):
        // Each WI in the SG reads B values and we construct int8 b
        // 
        // For each k-pair p (0..7):
        //   row0 = B[(k_base + 2*p) * N + b_col + sg_lid]  // half
        //   row1 = B[(k_base + 2*p + 1) * N + b_col + sg_lid]  // half
        //   b[p] = as_int((short2)(as_short(row0), as_short(row1)))

        // For A from SLM (row-major, stride=SLM_A_STRIDE):
        // For DPAS a (short8), we need A[row_base+r][k_base + sg_lid] for r=0..7
        // Using subgroup block read from SLM:
        //   base = &slm_A[row_base * SLM_A_STRIDE + k_base]
        //   stride between rows = SLM_A_STRIDE
        //   short8 a = intel_sub_group_block_read_us8((__local uint*)(base))
        //   This reads 8 consecutive ushorts starting at base + sg_lid * 2
        //   But we want stride = SLM_A_STRIDE between rows, not consecutive
        //   
        //   Actually intel_sub_group_block_read_us8 reads 8 consecutive ushort PER WI
        //   with layout: WI i gets elements at base + i, base + 16 + i, ..., base + 7*16 + i
        //   (stride = exec_size = 16 between vector elements)
        //   
        //   So if we store A in SLM with stride=16 (not 34), then:
        //   base = &slm_A[row_base * 16 + k_base]  (k_base = 0 or 16)
        //   intel_sub_group_block_read_us8 gives: WI i gets A[row_base + j][k_base + i] for j=0..7
        //   That's exactly what we need for short8 a!
        //   
        //   But stride=16 might cause bank conflicts. Let's try stride=16 first (no padding).
        //   Actually for SLM bank conflicts on Xe2: SLM has 32 banks, 4 bytes each
        //   With stride=16 halfs = 32 bytes: row r starts at byte 32*r
        //   Bank for element at byte offset b: bank = (b/4) % 32
        //   Row 0, element 0: bank 0; Row 1, element 0: bank 8; Row 2: bank 16; Row 3: bank 24
        //   Row 4: bank 0 again! So rows 0 and 4 conflict.
        //   
        //   With stride=18 halfs = 36 bytes: 
        //   Row 0: byte 0, bank 0; Row 1: byte 36, bank 9; Row 2: byte 72, bank 18; Row 3: byte 108, bank 27
        //   Row 4: byte 144, bank 4; Row 5: byte 180, bank 13; Row 6: byte 216, bank 22; Row 7: byte 252, bank 31
        //   No conflicts! But intel_sub_group_block_read_us8 assumes stride=16 (exec_size)...
        //   
        //   So we MUST use stride=16 for block reads to work. Let's accept potential bank conflicts
        //   or use manual loads.

        // Let me reconsider the SLM layout:
        // Store A as [32 rows × 32 K] with stride=32 (no padding needed for block reads if we
        // read 16 elements at a time with stride=16 between rows)
        // 
        // Actually, for intel_sub_group_block_read_us8:
        // It reads from a __local uint* pointer
        // The read pattern: 8 ushorts per WI, with stride = subgroup_size (16) between elements
        // So it reads a contiguous block of 8 * 16 = 128 ushorts = 256 bytes
        // Element [vec_idx][lane] is at offset (vec_idx * 16 + lane) * 2 bytes
        // 
        // For our A tile: we want A[row][k] where row=row_base..row_base+7, k=k_inner..k_inner+15
        // If SLM stride = 16 (halfs), then A[row][k] is at offset row*16 + k
        // block_read_us8 from base = &slm_A[row_base * 16 + k_inner]:
        //   vec_idx j, lane i → offset (j*16 + i) * 2 bytes → element at slm_A[row_base*16 + k_inner + j*16 + i]
        //   = slm_A[(row_base + j) * 16 + k_inner + i]
        //   = A[row_base + j][k_inner + i] ✓ (if k_inner is the start within the 16-element half of the 32-wide tile)
        //   
        // Wait, but our K-tile is 32 wide. If SLM stride = 16, we can only store 16 K-elements per row.
        // We need stride = 32 for the full K-tile.
        // 
        // With stride = 32: A[row][k] at offset row*32 + k
        // block_read_us8 from base = &slm_A[row_base * 32 + k_inner]:
        //   vec_idx j, lane i → element at offset row_base*32 + k_inner + j*16 + i
        //   For j=0, i=0..15: A[row_base][k_inner + 0..15] ✓
        //   For j=1, i=0..15: offset = row_base*32 + k_inner + 16 + i
        //     If k_inner=0: this is A[row_base][16..31] (second half of K)
        //     But we want A[row_base+1][0..15]!
        //   
        // So stride=32 doesn't work with block_read_us8 directly for reading 8 rows × 16 K.
        // We'd need stride=16 in SLM, meaning we store the 32×32 A tile as two 32×16 halves.

        // ALTERNATIVE: Store A in SLM transposed or in a DPAS-friendly layout
        // Store A as [K/16][rows][16] = [2][32][16] for our 32×32 tile
        // Then for k_half h (0 or 1), row_base r:
        //   base = &slm_A[h * 32 * 16 + r * 16]
        //   block_read_us8 gives 8 rows × 16 K-elements ✓

        // Let me use this layout:
        // SLM_A[k_half][row][k_within_half] where k_half=0,1; row=0..31; k_within_half=0..15
        // Linear offset: k_half * 512 + row * 16 + k_within_half
        // Total: 2 * 32 * 16 = 1024 halfs = 2048 bytes ✓

        // === K-inner loop: two steps of k=16 ===
        // Step 1: k_inner = 0..15
        {
            // Load A from SLM for rows 0-7 (for acc0)
            // A is at SLM offset: 0*512 + (row_offset)*16, stride between rows = 16
            // Use intel_sub_group_block_read_us8

            // Load B from global: B[k+0..k+15][b_col..b_col+15]
            // Pack into VNNI format (int8)

            // DPAS for acc0, acc1, acc2, acc3
        }
        // Step 2: k_inner = 16..31
        {
            // Similar with k_half = 1
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store results
    // Each SG writes 32 rows × 16 cols
    // ...
}
```

Console output from running this kernel:

Test result on platform Intel Corporation Battlemage G21 [Intel Graphics]:
==== test session starts

task.py::TestMatmulOCL::test_correctness_wrt_pytorch[0] FAILED           [ 25%]
task.py::TestMatmulOCL::test_correctness_wrt_pytorch[1] FAILED           [ 50%]
task.py::TestMatmulOCL::test_correctness_wrt_reference[0] FAILED         [ 75%]
task.py::TestMatmulOCL::test_correctness_wrt_reference[1] FAILED         [100%]

=================================== FAILURES ===================================
________________ TestMatmulOCL.test_correctness_wrt_pytorch[0] _________________

self = <task.TestMatmulOCL object at 0x7afab69c2d80>
kernel = <function initialize_matmul_kernel.<locals>.run at 0x7afaa893d1c0>
ocl_queue = <pyopencl._cl.CommandQueue object at 0x7afab6a3dfd0>, _run = 0

    @pytest.mark.parametrize("_run", range(2))
    def test_correctness_wrt_pytorch(self, kernel, ocl_queue, _run):
        args, expected = get_data_on_device(DEFAULT_SHAPE, ocl_queue, seed=_run)
        kernel(*args)

        _, _, out_buf, m, _, n = args
        got_flat = np.empty(int(m) * int(n), dtype=np.float16)
        cl.enqueue_copy(ocl_queue, got_flat, out_buf)
        got = got_flat.reshape((int(m), int(n))).astype(np.float32)

        assert got.shape == expected.shape
>       assert np.allclose(got, expected, rtol=2e-2, atol=2e-2), "matmul result mismatch vs pytorch/numpy"
E       AssertionError: matmul result mismatch vs pytorch/numpy
E       assert False
E        +  where False = <function allclose at 0x7afb0ef58e30>(array([[0., 0., 0., ..., 0., 0., 0.],\n       [0., 0., 0., ..., 0., 0., 0.],\n       [0., 0., 0., ..., 0., 0., 0.],\n       ...,\n       [0., 0., 0., ..., 0., 0., 0.],\n       [0., 0., 0., ..., 0., 0., 0.],\n       [0., 0., 0., ..., 0., 0., 0.]], shape=(2048, 2048), dtype=float32), array([[-12.434087 , -85.22102  ,  45.86866  , ..., -67.074715 ,\n        -64.52674  ,  37.798523 ],\n       [-82.95244  ,  28.332115 ,   4.3084497, ...,  37.17192  ,\n         48.87541  ,  55.1519   ],\n       [ 31.096529 , -51.77693  ,  -9.3054905, ...,   8.124319 ,\n         61.21928  ,   4.7092314],\n       ...,\n       [-65.29967  , -27.73106  ,  74.195465 , ..., 122.09403  ,\n        -41.569603 ,  10.711429 ],\n       [ 44.6838   ,   2.3142765,  22.61605  , ..., -35.807106 ,\n         42.793472 ,  52.60636  ],\n       [ 50.399834 ,  -3.015791 ,  21.545517 , ..., -21.399685 ,\n        -36.035267 ,  49.01544  ]], shape=(2048, 2048), dtype=float32), rtol=0.02, atol=0.02)
E        +    where <function allclose at 0x7afb0ef58e30> = np.allclose

task.py:244: AssertionError
________________ TestMatmulOCL.test_correctness_wrt_pytorch[1] _________________

self = <task.TestMatmulOCL object at 0x7afab697de80>
kernel = <function initialize_matmul_kernel.<locals>.run at 0x7afaa893d1c0>
ocl_queue = <pyopencl._cl.CommandQueue object at 0x7afab6a3dfd0>, _run = 1

    @pytest.mark.parametrize("_run", range(2))
    def test_correctness_wrt_pytorch(self, kernel, ocl_queue, _run):
        args, expected = get_data_on_device(DEFAULT_SHAPE, ocl_queue, seed=_run)
        kernel(*args)

        _, _, out_buf, m, _, n = args
        got_flat = np.empty(int(m) * int(n), dtype=np.float16)
        cl.enqueue_copy(ocl_queue, got_flat, out_buf)
        got = got_flat.reshape((int(m), int(n))).astype(np.float32)

        assert got.shape == expected.shape
>       assert np.allclose(got, expected, rtol=2e-2, atol=2e-2), "matmul result mismatch vs pytorch/numpy"
E       AssertionError: matmul result mismatch vs pytorch/numpy
E       assert False
E        +  where False = <function allclose at 0x7afb0ef58e30>(array([[0., 0., 0., ..., 0., 0., 0.],\n       [0., 0., 0., ..., 0., 0., 0.],\n       [0., 0., 0., ..., 0., 0., 0.],\n       ...,\n       [0., 0., 0., ..., 0., 0., 0.],\n       [0., 0., 0., ..., 0., 0., 0.],\n       [0., 0., 0., ..., 0., 0., 0.]], shape=(2048, 2048), dtype=float32), array([[  63.220627 ,   13.663691 ,   62.708282 , ...,   26.950535 ,\n        -100.14888  ,  -76.10468  ],\n       [  30.338015 ,   -9.576593 ,  -15.848044 , ...,  -86.66203  ,\n           6.3691177,    9.569207 ],\n       [  -9.825886 ,   26.83852  ,  -39.88768  , ...,   94.32298  ,\n         -40.437588 ,   13.349518 ],\n       ...,\n       [ -50.946926 ,  -10.7210655,  -18.652342 , ...,   -4.0612535,\n         -29.112085 ,   -2.7683525],\n       [ -41.46417  ,   -5.034666 ,   35.500336 , ...,    3.5289268,\n          14.26104  ,   55.58531  ],\n       [ -33.896618 ,   51.45737  ,   13.108513 , ...,   11.92079  ,\n         -64.022385 ,   63.048595 ]], shape=(2048, 2048), dtype=float32), rtol=0.02, atol=0.02)
E        +    where <function allclose at 0x7afb0ef58e30> = np.allclose

task.py:244: AssertionError
_______________ TestMatmulOCL.test_correctness_wrt_reference[0] ________________

self = <task.TestMatmulOCL object at 0x7afab6a2b860>
kernel = <function initialize_matmul_kernel.<locals>.run at 0x7afaa893d1c0>
ocl_queue = <pyopencl._cl.CommandQueue object at 0x7afab6a3dfd0>, _run = 0

    @pytest.mark.parametrize("_run", range(2))
    def test_correctness_wrt_reference(self, kernel, ocl_queue, _run):
        args, _ = get_data_on_device(DEFAULT_SHAPE, ocl_queue, seed=100 + _run)
        kernel(*args)

        _, _, out_buf, m, _, n = args
        got_flat = np.empty(int(m) * int(n), dtype=np.float16)
        cl.enqueue_copy(ocl_queue, got_flat, out_buf)
        got = got_flat.reshape((int(m), int(n))).astype(np.float32)

        ref_kernel = initialize_matmul_kernel("matmul_reference.cl", ocl_queue)
        ref_kernel(*args)
        ref_flat = np.empty(int(m) * int(n), dtype=np.float16)
        cl.enqueue_copy(ocl_queue, ref_flat, out_buf)
        ref = ref_flat.reshape((int(m), int(n))).astype(np.float32)

>       assert np.allclose(got, ref, rtol=2e-2, atol=2e-2), "matmul result mismatch vs reference"
E       AssertionError: matmul result mismatch vs reference
E       assert False
E        +  where False = <function allclose at 0x7afb0ef58e30>(array([[0., 0., 0., ..., 0., 0., 0.],\n       [0., 0., 0., ..., 0., 0., 0.],\n       [0., 0., 0., ..., 0., 0., 0.],\n       ...,\n       [0., 0., 0., ..., 0., 0., 0.],\n       [0., 0., 0., ..., 0., 0., 0.],\n       [0., 0., 0., ..., 0., 0., 0.]], shape=(2048, 2048), dtype=float32), array([[-5.7125000e+01, -2.8015625e+01,  6.5562500e+01, ...,\n        -5.0218750e+01,  2.7000000e+01,  2.6109375e+01],\n       [ 9.2687500e+01,  1.8343750e+01, -1.4343750e+01, ...,\n        -4.0843750e+01, -2.5828125e+01, -3.1914062e+00],\n       [-2.5585938e+00, -3.0015625e+01,  4.9937500e+01, ...,\n        -5.7156250e+01, -9.2250000e+01,  1.2921875e+01],\n       ...,\n       [-4.6031250e+01, -1.4262500e+02, -6.1500000e+01, ...,\n        -1.2194824e-01,  6.6445312e+00, -5.3156250e+01],\n       [ 3.1859375e+01, -2.3484375e+01,  4.2750000e+01, ...,\n         1.8390625e+01, -2.0507812e+00, -1.6000000e+02],\n       [-3.8968750e+01,  9.1750000e+01,  2.3953125e+01, ...,\n         4.5437500e+01,  5.4437500e+01,  1.0787500e+02]],\n      shape=(2048, 2048), dtype=float32), rtol=0.02, atol=0.02)
E        +    where <function allclose at 0x7afb0ef58e30> = np.allclose

task.py:262: AssertionError
_______________ TestMatmulOCL.test_correctness_wrt_reference[1] ________________

self = <task.TestMatmulOCL object at 0x7afab6a2baa0>
kernel = <function initialize_matmul_kernel.<locals>.run at 0x7afaa893d1c0>
ocl_queue = <pyopencl._cl.CommandQueue object at 0x7afab6a3dfd0>, _run = 1

    @pytest.mark.parametrize("_run", range(2))
    def test_correctness_wrt_reference(self, kernel, ocl_queue, _run):
        args, _ = get_data_on_device(DEFAULT_SHAPE, ocl_queue, seed=100 + _run)
        kernel(*args)

        _, _, out_buf, m, _, n = args
        got_flat = np.empty(int(m) * int(n), dtype=np.float16)
        cl.enqueue_copy(ocl_queue, got_flat, out_buf)
        got = got_flat.reshape((int(m), int(n))).astype(np.float32)

        ref_kernel = initialize_matmul_kernel("matmul_reference.cl", ocl_queue)
        ref_kernel(*args)
        ref_flat = np.empty(int(m) * int(n), dtype=np.float16)
        cl.enqueue_copy(ocl_queue, ref_flat, out_buf)
        ref = ref_flat.reshape((int(m), int(n))).astype(np.float32)

>       assert np.allclose(got, ref, rtol=2e-2, atol=2e-2), "matmul result mismatch vs reference"
E       AssertionError: matmul result mismatch vs reference
E       assert False
E        +  where False = <function allclose at 0x7afb0ef58e30>(array([[0., 0., 0., ..., 0., 0., 0.],\n       [0., 0., 0., ..., 0., 0., 0.],\n       [0., 0., 0., ..., 0., 0., 0.],\n       ...,\n       [0., 0., 0., ..., 0., 0., 0.],\n       [0., 0., 0., ..., 0., 0., 0.],\n       [0., 0., 0., ..., 0., 0., 0.]], shape=(2048, 2048), dtype=float32), array([[  46.4375   ,  -21.75     ,  -36.71875  , ...,  -39.96875  ,\n           3.125    , -147.25     ],\n       [ -13.859375 ,  -17.453125 ,   17.375    , ...,   57.90625  ,\n          25.859375 ,  -68.0625   ],\n       [  34.1875   ,   -9.421875 ,  -49.       , ...,  -46.71875  ,\n         -64.5625   ,   37.6875   ],\n       ...,\n       [   3.3261719,  -60.03125  ,  -64.       , ...,   18.328125 ,\n         -17.234375 ,  -30.90625  ],\n       [  -0.9145508,   97.25     ,  -18.109375 , ...,   35.75     ,\n         -45.0625   ,   21.78125  ],\n       [ -36.5      ,  -63.8125   ,   68.125    , ...,  -48.09375  ,\n         -29.515625 ,   11.1640625]], shape=(2048, 2048), dtype=float32), rtol=0.02, atol=0.02)
E        +    where <function allclose at 0x7afb0ef58e30> = np.allclose

task.py:262: AssertionError
=============================== warnings summary ===============================
task.py::TestMatmulOCL::test_correctness_wrt_pytorch[0]
  /home/openvino-ci-74/miniforge3/envs/kernel_intel/lib/python3.12/site-packages/pyopencl/cache.py:517: CompilerWarning: Non-empty compiler output encountered. Set the environment variable PYOPENCL_COMPILER_OUTPUT=1 to see more.
    _create_built_program_from_source_cached(

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ============================
FAILED task.py::TestMatmulOCL::test_correctness_wrt_pytorch[0] - AssertionErr...
FAILED task.py::TestMatmulOCL::test_correctness_wrt_pytorch[1] - AssertionErr...
FAILED task.py::TestMatmulOCL::test_correctness_wrt_reference[0] - AssertionE...
FAILED task.py::TestMatmulOCL::test_correctness_wrt_reference[1] - AssertionE...
================== 4 failed, 1 deselected, 1 warning in 0.87s ==================

## Hardware specification:
Your code will run on the following hardware:
**Intel Battlemage** with specs: Xe-cores: 20, Render Slices: 5, Ray Tracing Units: 20, Intel® Xe Matrix Extensions (Intel® XMX) Engines: 160, Xe Vector Engines: 160, Graphics Clock: 2670, GPU Peak TOPS (Int8): 233, TBP: 190, PCI Express Configurations ‡: PCI Express 4.0 x8, Device ID: 0xE20B, Memory: 12 GB GDDR6, Memory Interface: 192 bit, Memory Bandwidth: 456, Memory Speed: 19, ISA_GPU: Xe2-HPG
Please consider the hardware specifications when improving the code. 

## Task:

**Your objectives**:
1. Analyze the previous kernel and its evaluation log.
2. Identify any errors or mismatches with the reference implementation.
3. Propose specific improvements or fixes, explaining your reasoning step by step.
4. Rewrite the kernel, providing the complete, corrected code in a code block.

**Critical Requirements:**

1. The kernel must exactly match the reference implementation's functionality.
2. The code must compile and run properly on the GPU.
3. Do not cache or reuse previous results; ensure the code executes fully on each run.
4. Keep all hyperparameters (e.g., batch size, dimensions) unchanged as specified in the reference implementation.
8. Beware of the critical error "Unexpected kernel lambda size. In such cases removing constexpr specifier aligns the captures between the host compiler and the device compiler"! Do not capture constexpr variables in lambda functions passed to kernel launches as this can lead to different lambda sizes between the host and device compiler.

Additional Guidance:

1. Clearly comment on any fixes and optimizations you implement.
2. If you change the kernel structure, explain why.
3. Anticipate and address possible runtime or compilation errors.

Please structure your response as follows:

1. Analysis:
    * Summarize the errors found in the previous kernel and log.
    * Explain your proposed changes.
2. Improved OCL code:
    * Provide the complete, corrected OCL code in a code block:
```OCL
Your code here
```


## Required Optimizations

Apply the following optimization techniques in your implementation:

- **Kernel Fusion**: Combine sequential operations (e.g., exp → add → activation) into a single kernel. Eliminate intermediate buffers by computing in registers.
- **Sub-Group Collectives**: Use `reduce_over_group(sg, val, op)` for hardware-accelerated SIMD reductions. Use `group_broadcast` and `shift_group_*` for efficient data sharing.