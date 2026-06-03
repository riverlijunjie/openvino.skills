

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

### Version 2 (Result: Correctness score: 5 / 5 (compiles and correct), runtime in ms: 1.080):
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
    const int sg_id = get_local_id(1);       // 0..3
    const int sg_lid = get_sub_group_local_id(); // 0..15

    const int wg_row = get_group_id(1) * 32;
    const int wg_col = get_group_id(0) * 64;
    const int sg_col = wg_col + sg_id * 16;

    // SLM layout: two K16 blocks per buffer, stride=16 for block reads
    // Use stride=18 to reduce bank conflicts (16 lanes * 2 bytes = 32 bytes per row,
    // with stride=16 all rows hit same banks; stride=18 shifts by 4 bytes per row)
    #define SLM_K16_STRIDE 16
    #define SLM_BUF_SIZE 1024  // 32 rows * 32 cols = 1024 halfs per buffer
    __local half slm_A[2 * SLM_BUF_SIZE];

    float8 acc0 = 0.0f;
    float8 acc1 = 0.0f;
    float8 acc2 = 0.0f;
    float8 acc3 = 0.0f;

    const int lid = sg_id * 16 + sg_lid;
    const int a_load_row = lid / 2;
    const int a_load_khalf = lid & 1;

    // Preload first A tile into SLM buffer 0
    {
        __global const half* a_ptr = A + (wg_row + a_load_row) * K + a_load_khalf * 16;
        int slm_off = a_load_khalf * (32 * SLM_K16_STRIDE) + a_load_row * SLM_K16_STRIDE;
        half8 v0 = vload8(0, a_ptr);
        half8 v1 = vload8(1, a_ptr);
        vstore8(v0, 0, &slm_A[slm_off]);
        vstore8(v1, 0, &slm_A[slm_off + 8]);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    const int k_iterations = K / 32;  // K guaranteed divisible by 32

    // Main K-loop: process one K=32 tile per iteration with double-buffered A
    // Unroll 2x to reduce loop overhead and barrier cost
    int buf = 0;

    // Process pairs of K-tiles (2x unroll)
    const int k_pairs = k_iterations / 2;
    const int k_remainder_iters = k_iterations & 1;

    for (int kp = 0; kp < k_pairs; kp++) {
        int k_offset = kp * 64;

        // ============ FIRST K=32 TILE ============
        {
            int slm_base = buf * SLM_BUF_SIZE;
            __global const half* b_base = B + k_offset * N + sg_col + sg_lid;

            // --- K16 step 0 ---
            short8 a0, a1, a2, a3;
            int a_off = slm_base;
            a0 = as_short8(intel_sub_group_block_read_us8((__local const uint*)&slm_A[a_off]));
            a1 = as_short8(intel_sub_group_block_read_us8((__local const uint*)&slm_A[a_off + 8*SLM_K16_STRIDE]));
            a2 = as_short8(intel_sub_group_block_read_us8((__local const uint*)&slm_A[a_off + 16*SLM_K16_STRIDE]));
            a3 = as_short8(intel_sub_group_block_read_us8((__local const uint*)&slm_A[a_off + 24*SLM_K16_STRIDE]));

            int8 b_reg;
            __global const half* bp = b_base;
            b_reg.s0 = as_int((half2)(bp[0], bp[N]));
            b_reg.s1 = as_int((half2)(bp[2*N], bp[3*N]));
            b_reg.s2 = as_int((half2)(bp[4*N], bp[5*N]));
            b_reg.s3 = as_int((half2)(bp[6*N], bp[7*N]));
            b_reg.s4 = as_int((half2)(bp[8*N], bp[9*N]));
            b_reg.s5 = as_int((half2)(bp[10*N], bp[11*N]));
            b_reg.s6 = as_int((half2)(bp[12*N], bp[13*N]));
            b_reg.s7 = as_int((half2)(bp[14*N], bp[15*N]));

            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b_reg, acc0);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b_reg, acc1);
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b_reg, acc2);
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b_reg, acc3);

            // --- K16 step 1 ---
            a_off = slm_base + 32 * SLM_K16_STRIDE;
            a0 = as_short8(intel_sub_group_block_read_us8((__local const uint*)&slm_A[a_off]));
            a1 = as_short8(intel_sub_group_block_read_us8((__local const uint*)&slm_A[a_off + 8*SLM_K16_STRIDE]));
            a2 = as_short8(intel_sub_group_block_read_us8((__local const uint*)&slm_A[a_off + 16*SLM_K16_STRIDE]));
            a3 = as_short8(intel_sub_group_block_read_us8((__local const uint*)&slm_A[a_off + 24*SLM_K16_STRIDE]));

            bp = b_base + 16 * N;
            b_reg.s0 = as_int((half2)(bp[0], bp[N]));
            b_reg.s1 = as_int((half2)(bp[2*N], bp[3*N]));
            b_reg.s2 = as_int((half2)(bp[4*N], bp[5*N]));
            b_reg.s3 = as_int((half2)(bp[6*N], bp[7*N]));
            b_reg.s4 = as_int((half2)(bp[8*N], bp[9*N]));
            b_reg.s5 = as_int((half2)(bp[10*N], bp[11*N]));
            b_reg.s6 = as_int((half2)(bp[12*N], bp[13*N]));
            b_reg.s7 = as_int((half2)(bp[14*N], bp[15*N]));

            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b_reg, acc0);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b_reg, acc1);
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b_reg, acc2);
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b_reg, acc3);
        }

        // Load next A tile (for second K=32 tile)
        {
            int next_buf = 1 - buf;
            int next_k = k_offset + 32;
            __global const half* a_ptr = A + (wg_row + a_load_row) * K + next_k + a_load_khalf * 16;
            int slm_off = next_buf * SLM_BUF_SIZE + a_load_khalf * (32 * SLM_K16_STRIDE) + a_load_row * SLM_K16_STRIDE;
            half8 v0 = vload8(0, a_ptr);
            half8 v1 = vload8(1, a_ptr);
            vstore8(v0, 0, &slm_A[slm_off]);
            vstore8(v1, 0, &slm_A[slm_off + 8]);
            buf = next_buf;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // ============ SECOND K=32 TILE ============
        {
            int slm_base = buf * SLM_BUF_SIZE;
            __global const half* b_base = B + (k_offset + 32) * N + sg_col + sg_lid;

            // --- K16 step 0 ---
            short8 a0, a1, a2, a3;
            int a_off = slm_base;
            a0 = as_short8(intel_sub_group_block_read_us8((__local const uint*)&slm_A[a_off]));
            a1 = as_short8(intel_sub_group_block_read_us8((__local const uint*)&slm_A[a_off + 8*SLM_K16_STRIDE]));
            a2 = as_short8(intel_sub_group_block_read_us8((__local const uint*)&slm_A[a_off + 16*SLM_K16_STRIDE]));
            a3 = as_short8(intel_sub_group_block_read_us8((__local const uint*)&slm_A[a_off + 24*SLM_K16_STRIDE]));

            int8 b_reg;
            __global const half* bp = b_base;
            b_reg.s0 = as_int((half2)(bp[0], bp[N]));
            b_reg.s1 = as_int((half2)(bp[2*N], bp[3*N]));
            b_reg.s2 = as_int((half2)(bp[4*N], bp[5*N]));
            b_reg.s3 = as_int((half2)(bp[6*N], bp[7*N]));
            b_reg.s4 = as_int((half2)(bp[8*N], bp[9*N]));
            b_reg.s5 = as_int((half2)(bp[10*N], bp[11*N]));
            b_reg.s6 = as_int((half2)(bp[12*N], bp[13*N]));
            b_reg.s7 = as_int((half2)(bp[14*N], bp[15*N]));

            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b_reg, acc0);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b_reg, acc1);
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b_reg, acc2);
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b_reg, acc3);

            // --- K16 step 1 ---
            a_off = slm_base + 32 * SLM_K16_STRIDE;
            a0 = as_short8(intel_sub_group_block_read_us8((__local const uint*)&slm_A[a_off]));
            a1 = as_short8(intel_sub_group_block_read_us8((__local const uint*)&slm_A[a_off + 8*SLM_K16_STRIDE]));
            a2 = as_short8(intel_sub_group_block_read_us8((__local const uint*)&slm_A[a_off + 16*SLM_K16_STRIDE]));
            a3 = as_short8(intel_sub_group_block_read_us8((__local const uint*)&slm_A[a_off + 24*SLM_K16_STRIDE]));

            bp = b_base + 16 * N;
            b_reg.s0 = as_int((half2)(bp[0], bp[N]));
            b_reg.s1 = as_int((half2)(bp[2*N], bp[3*N]));
            b_reg.s2 = as_int((half2)(bp[4*N], bp[5*N]));
            b_reg.s3 = as_int((half2)(bp[6*N], bp[7*N]));
            b_reg.s4 = as_int((half2)(bp[8*N], bp[9*N]));
            b_reg.s5 = as_int((half2)(bp[10*N], bp[11*N]));
            b_reg.s6 = as_int((half2)(bp[12*N], bp[13*N]));
            b_reg.s7 = as_int((half2)(bp[14*N], bp[15*N]));

            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b_reg, acc0);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b_reg, acc1);
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b_reg, acc2);
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b_reg, acc3);
        }

        // Load next A tile (for next pair's first tile)
        {
            int next_buf = 1 - buf;
            int next_k = k_offset + 64;
            if (next_k < K) {
                __global const half* a_ptr = A + (wg_row + a_load_row) * K + next_k + a_load_khalf * 16;
                int slm_off = next_buf * SLM_BUF_SIZE + a_load_khalf * (32 * SLM_K16_STRIDE) + a_load_row * SLM_K16_STRIDE;
                half8 v0 = vload8(0, a_ptr);
                half8 v1 = vload8(1, a_ptr);
                vstore8(v0, 0, &slm_A[slm_off]);
                vstore8(v1, 0, &slm_A[slm_off + 8]);
            }
            buf = next_buf;
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Handle remaining single K=32 tile if k_iterations is odd
    if (k_remainder_iters) {
        int k_offset = k_pairs * 64;
        int slm_base = buf * SLM_BUF_SIZE;
        __global const half* b_base = B + k_offset * N + sg_col + sg_lid;

        // K16 step 0
        short8 a0, a1, a2, a3;
        int a_off = slm_base;
        a0 = as_short8(intel_sub_group_block_read_us8((__local const uint*)&slm_A[a_off]));
        a1 = as_short8(intel_sub_group_block_read_us8((__local const uint*)&slm_A[a_off + 8*SLM_K16_STRIDE]));
        a2 = as_short8(intel_sub_group_block_read_us8((__local const uint*)&slm_A[a_off + 16*SLM_K16_STRIDE]));
        a3 = as_short8(intel_sub_group_block_read_us8((__local const uint*)&slm_A[a_off + 24*SLM_K16_STRIDE]));

        int8 b_reg;
        __global const half* bp = b_base;
        b_reg.s0 = as_int((half2)(bp[0], bp[N]));
        b_reg.s1 = as_int((half2)(bp[2*N], bp[3*N]));
        b_reg.s2 = as_int((half2)(bp[4*N], bp[5*N]));
        b_reg.s3 = as_int((half2)(bp[6*N], bp[7*N]));
        b_reg.s4 = as_int((half2)(bp[8*N], bp[9*N]));
        b_reg.s5 = as_int((half2)(bp[10*N], bp[11*N]));
        b_reg.s6 = as_int((half2)(bp[12*N], bp[13*N]));
        b_reg.s7 = as_int((half2)(bp[14*N], bp[15*N]));

        acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b_reg, acc0);
        acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b_reg, acc1);
        acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b_reg, acc2);
        acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b_reg, acc3);

        // K16 step 1
        a_off = slm_base + 32 * SLM_K16_STRIDE;
        a0 = as_short8(intel_sub_group_block_read_us8((__local const uint*)&slm_A[a_off]));
        a1 = as_short8(intel_sub_group_block_read_us8((__local const uint*)&slm_A[a_off + 8*SLM_K16_STRIDE]));
        a2 = as_short8(intel_sub_group_block_read_us8((__local const uint*)&slm_A[a_off + 16*SLM_K16_STRIDE]));
        a3 = as_short8(intel_sub_group_block_read_us8((__local const uint*)&slm_A[a_off + 24*SLM_K16_STRIDE]));

        bp = b_base + 16 * N;
        b_reg.s0 = as_int((half2)(bp[0], bp[N]));
        b_reg.s1 = as_int((half2)(bp[2*N], bp[3*N]));
        b_reg.s2 = as_int((half2)(bp[4*N], bp[5*N]));
        b_reg.s3 = as_int((half2)(bp[6*N], bp[7*N]));
        b_reg.s4 = as_int((half2)(bp[8*N], bp[9*N]));
        b_reg.s5 = as_int((half2)(bp[10*N], bp[11*N]));
        b_reg.s6 = as_int((half2)(bp[12*N], bp[13*N]));
        b_reg.s7 = as_int((half2)(bp[14*N], bp[15*N]));

        acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b_reg, acc0);
        acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b_reg, acc1);
        acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b_reg, acc2);
        acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b_reg, acc3);
    }

    // === Store C using subgroup block writes ===
    // Each subgroup writes 32 rows x 16 cols
    // Use intel_sub_group_block_write_us for vectorized 16-wide stores
    __global half* c_base = C + wg_row * N + sg_col;

    // Convert and store using block writes (each writes 16 halfs = one row segment)
    intel_sub_group_block_write_us((__global uint*)(c_base + 0*N), as_ushort(convert_half(acc0.s0)));
    intel_sub_group_block_write_us((__global uint*)(c_base + 1*N), as_ushort(convert_half(acc0.s1)));
    intel_sub_group_block_write_us((__global uint*)(c_base + 2*N), as_ushort(convert_half(acc0.s2)));
    intel_sub_group_block_write_us((__global uint*)(c_base + 3*N), as_ushort(convert_half(acc0.s3)));
    intel_sub_group_block_write_us((__global uint*)(c_base + 4*N), as_ushort(convert_half(acc0.s4)));
    intel_sub_group_block_write_us((__global uint*)(c_base + 5*N), as_ushort(convert_half(acc0.s5)));
    intel_sub_group_block_write_us((__global uint*)(c_base + 6*N), as_ushort(convert_half(acc0.s6)));
    intel_sub_group_block_write_us((__global uint*)(c_base + 7*N), as_ushort(convert_half(acc0.s7)));

    intel_sub_group_block_write_us((__global uint*)(c_base + 8*N), as_ushort(convert_half(acc1.s0)));
    intel_sub_group_block_write_us((__global uint*)(c_base + 9*N), as_ushort(convert_half(acc1.s1)));
    intel_sub_group_block_write_us((__global uint*)(c_base + 10*N), as_ushort(convert_half(acc1.s2)));
    intel_sub_group_block_write_us((__global uint*)(c_base + 11*N), as_ushort(convert_half(acc1.s3)));
    intel_sub_group_block_write_us((__global uint*)(c_base + 12*N), as_ushort(convert_half(acc1.s4)));
    intel_sub_group_block_write_us((__global uint*)(c_base + 13*N), as_ushort(convert_half(acc1.s5)));
    intel_sub_group_block_write_us((__global uint*)(c_base + 14*N), as_ushort(convert_half(acc1.s6)));
    intel_sub_group_block_write_us((__global uint*)(c_base + 15*N), as_ushort(convert_half(acc1.s7)));

    intel_sub_group_block_write_us((__global uint*)(c_base + 16*N), as_ushort(convert_half(acc2.s0)));
    intel_sub_group_block_write_us((__global uint*)(c_base + 17*N), as_ushort(convert_half(acc2.s1)));
    intel_sub_group_block_write_us((__global uint*)(c_base + 18*N), as_ushort(convert_half(acc2.s2)));
    intel_sub_group_block_write_us((__global uint*)(c_base + 19*N), as_ushort(convert_half(acc2.s3)));
    intel_sub_group_block_write_us((__global uint*)(c_base + 20*N), as_ushort(convert_half(acc2.s4)));
    intel_sub_group_block_write_us((__global uint*)(c_base + 21*N), as_ushort(convert_half(acc2.s5)));
    intel_sub_group_block_write_us((__global uint*)(c_base + 22*N), as_ushort(convert_half(acc2.s6)));
    intel_sub_group_block_write_us((__global uint*)(c_base + 23*N), as_ushort(convert_half(acc2.s7)));

    intel_sub_group_block_write_us((__global uint*)(c_base + 24*N), as_ushort(convert_half(acc3.s0)));
    intel_sub_group_block_write_us((__global uint*)(c_base + 25*N), as_ushort(convert_half(acc3.s1)));
    intel_sub_group_block_write_us((__global uint*)(c_base + 26*N), as_ushort(convert_half(acc3.s2)));
    intel_sub_group_block_write_us((__global uint*)(c_base + 27*N), as_ushort(convert_half(acc3.s3)));
    intel_sub_group_block_write_us((__global uint*)(c_base + 28*N), as_ushort(convert_half(acc3.s4)));
    intel_sub_group_block_write_us((__global uint*)(c_base + 29*N), as_ushort(convert_half(acc3.s5)));
    intel_sub_group_block_write_us((__global uint*)(c_base + 30*N), as_ushort(convert_half(acc3.s6)));
    intel_sub_group_block_write_us((__global uint*)(c_base + 31*N), as_ushort(convert_half(acc3.s7)));
}
```

### Previous kernel and evaluation:
Here is the last OCL kernel we tested (Result: Correctness score: 5 / 5 (compiles and correct), runtime in ms: 1.020):
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
    const int sg_id = get_local_id(1);       // 0..3
    const int sg_lid = get_sub_group_local_id(); // 0..15

    const int wg_row = get_group_id(1) * 32;
    const int wg_col = get_group_id(0) * 64;
    const int sg_col = wg_col + sg_id * 16;

    // SLM: double-buffered, layout optimized for intel_sub_group_block_read_us8
    // Each buffer: two K16 blocks, each 32 rows x 16 cols, stride=16
    // Total: 2 * 32 * 32 = 2048 halfs = 4096 bytes
    #define SLM_K16_STRIDE 16
    #define SLM_BUF_SIZE 1024
    __local half slm_A[2 * SLM_BUF_SIZE];

    float8 acc0 = 0.0f;
    float8 acc1 = 0.0f;
    float8 acc2 = 0.0f;
    float8 acc3 = 0.0f;

    const int lid = sg_id * 16 + sg_lid;
    const int a_load_row = lid / 2;
    const int a_load_khalf = lid & 1;

    // Precompute A load base
    const int a_row_stride = K;  // stride between rows in A

    // Preload first A tile into SLM buffer 0
    {
        __global const half* a_ptr = A + (wg_row + a_load_row) * a_row_stride + a_load_khalf * 16;
        int slm_off = a_load_khalf * (32 * SLM_K16_STRIDE) + a_load_row * SLM_K16_STRIDE;

        half8 v0 = vload8(0, a_ptr);
        half8 v1 = vload8(1, a_ptr);
        vstore8(v0, 0, &slm_A[slm_off]);
        vstore8(v1, 0, &slm_A[slm_off + 8]);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    int buf = 0;
    const int k_iterations = K / 32;  // K guaranteed divisible by 32

    // Main K-loop - no remainder handling needed
    for (int ki = 0; ki < k_iterations - 1; ki++) {
        int k_offset = ki * 32;
        int next_k = k_offset + 32;
        int next_buf = 1 - buf;
        int slm_base = buf * SLM_BUF_SIZE;

        // B pointer for this K-step
        __global const half* b_base = B + k_offset * N + sg_col + sg_lid;

        // --- K16 step 0: Load A from SLM, Load B from global, DPAS ---
        short8 a0_0, a1_0, a2_0, a3_0;
        {
            int a_off = slm_base;
            a0_0 = as_short8(intel_sub_group_block_read_us8((__local const uint*)&slm_A[a_off + 0*SLM_K16_STRIDE]));
            a1_0 = as_short8(intel_sub_group_block_read_us8((__local const uint*)&slm_A[a_off + 8*SLM_K16_STRIDE]));
            a2_0 = as_short8(intel_sub_group_block_read_us8((__local const uint*)&slm_A[a_off + 16*SLM_K16_STRIDE]));
            a3_0 = as_short8(intel_sub_group_block_read_us8((__local const uint*)&slm_A[a_off + 24*SLM_K16_STRIDE]));
        }

        int8 b_reg0;
        {
            __global const half* bp = b_base;
            b_reg0.s0 = as_int((half2)(bp[0*N], bp[1*N]));
            b_reg0.s1 = as_int((half2)(bp[2*N], bp[3*N]));
            b_reg0.s2 = as_int((half2)(bp[4*N], bp[5*N]));
            b_reg0.s3 = as_int((half2)(bp[6*N], bp[7*N]));
            b_reg0.s4 = as_int((half2)(bp[8*N], bp[9*N]));
            b_reg0.s5 = as_int((half2)(bp[10*N], bp[11*N]));
            b_reg0.s6 = as_int((half2)(bp[12*N], bp[13*N]));
            b_reg0.s7 = as_int((half2)(bp[14*N], bp[15*N]));
        }

        acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0_0, b_reg0, acc0);
        acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1_0, b_reg0, acc1);

        // Start loading next A tile into other buffer (interleaved with compute)
        __global const half* a_next_ptr = A + (wg_row + a_load_row) * a_row_stride + next_k + a_load_khalf * 16;
        half8 a_next_v0 = vload8(0, a_next_ptr);
        half8 a_next_v1 = vload8(1, a_next_ptr);

        acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2_0, b_reg0, acc2);
        acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3_0, b_reg0, acc3);

        // --- K16 step 1 ---
        short8 a0_1, a1_1, a2_1, a3_1;
        {
            int a_off = slm_base + 32 * SLM_K16_STRIDE;
            a0_1 = as_short8(intel_sub_group_block_read_us8((__local const uint*)&slm_A[a_off + 0*SLM_K16_STRIDE]));
            a1_1 = as_short8(intel_sub_group_block_read_us8((__local const uint*)&slm_A[a_off + 8*SLM_K16_STRIDE]));
            a2_1 = as_short8(intel_sub_group_block_read_us8((__local const uint*)&slm_A[a_off + 16*SLM_K16_STRIDE]));
            a3_1 = as_short8(intel_sub_group_block_read_us8((__local const uint*)&slm_A[a_off + 24*SLM_K16_STRIDE]));
        }

        int8 b_reg1;
        {
            __global const half* bp = b_base + 16 * N;
            b_reg1.s0 = as_int((half2)(bp[0*N], bp[1*N]));
            b_reg1.s1 = as_int((half2)(bp[2*N], bp[3*N]));
            b_reg1.s2 = as_int((half2)(bp[4*N], bp[5*N]));
            b_reg1.s3 = as_int((half2)(bp[6*N], bp[7*N]));
            b_reg1.s4 = as_int((half2)(bp[8*N], bp[9*N]));
            b_reg1.s5 = as_int((half2)(bp[10*N], bp[11*N]));
            b_reg1.s6 = as_int((half2)(bp[12*N], bp[13*N]));
            b_reg1.s7 = as_int((half2)(bp[14*N], bp[15*N]));
        }

        acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0_1, b_reg1, acc0);
        acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1_1, b_reg1, acc1);
        acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2_1, b_reg1, acc2);
        acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3_1, b_reg1, acc3);

        // Store next A tile to SLM
        int slm_off = next_buf * SLM_BUF_SIZE + a_load_khalf * (32 * SLM_K16_STRIDE) + a_load_row * SLM_K16_STRIDE;
        vstore8(a_next_v0, 0, &slm_A[slm_off]);
        vstore8(a_next_v1, 0, &slm_A[slm_off + 8]);

        buf = next_buf;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Last iteration - no next A load needed
    {
        int k_offset = (k_iterations - 1) * 32;
        int slm_base = buf * SLM_BUF_SIZE;
        __global const half* b_base = B + k_offset * N + sg_col + sg_lid;

        // K16 step 0
        short8 a0_0, a1_0, a2_0, a3_0;
        {
            int a_off = slm_base;
            a0_0 = as_short8(intel_sub_group_block_read_us8((__local const uint*)&slm_A[a_off + 0*SLM_K16_STRIDE]));
            a1_0 = as_short8(intel_sub_group_block_read_us8((__local const uint*)&slm_A[a_off + 8*SLM_K16_STRIDE]));
            a2_0 = as_short8(intel_sub_group_block_read_us8((__local const uint*)&slm_A[a_off + 16*SLM_K16_STRIDE]));
            a3_0 = as_short8(intel_sub_group_block_read_us8((__local const uint*)&slm_A[a_off + 24*SLM_K16_STRIDE]));
        }

        int8 b_reg0;
        {
            __global const half* bp = b_base;
            b_reg0.s0 = as_int((half2)(bp[0*N], bp[1*N]));
            b_reg0.s1 = as_int((half2)(bp[2*N], bp[3*N]));
            b_reg0.s2 = as_int((half2)(bp[4*N], bp[5*N]));
            b_reg0.s3 = as_int((half2)(bp[6*N], bp[7*N]));
            b_reg0.s4 = as_int((half2)(bp[8*N], bp[9*N]));
            b_reg0.s5 = as_int((half2)(bp[10*N], bp[11*N]));
            b_reg0.s6 = as_int((half2)(bp[12*N], bp[13*N]));
            b_reg0.s7 = as_int((half2)(bp[14*N], bp[15*N]));
        }

        acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0_0, b_reg0, acc0);
        acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1_0, b_reg0, acc1);
        acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2_0, b_reg0, acc2);
        acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3_0, b_reg0, acc3);

        // K16 step 1
        short8 a0_1, a1_1, a2_1, a3_1;
        {
            int a_off = slm_base + 32 * SLM_K16_STRIDE;
            a0_1 = as_short8(intel_sub_group_block_read_us8((__local const uint*)&slm_A[a_off + 0*SLM_K16_STRIDE]));
            a1_1 = as_short8(intel_sub_group_block_read_us8((__local const uint*)&slm_A[a_off + 8*SLM_K16_STRIDE]));
            a2_1 = as_short8(intel_sub_group_block_read_us8((__local const uint*)&slm_A[a_off + 16*SLM_K16_STRIDE]));
            a3_1 = as_short8(intel_sub_group_block_read_us8((__local const uint*)&slm_A[a_off + 24*SLM_K16_STRIDE]));
        }

        int8 b_reg1;
        {
            __global const half* bp = b_base + 16 * N;
            b_reg1.s0 = as_int((half2)(bp[0*N], bp[1*N]));
            b_reg1.s1 = as_int((half2)(bp[2*N], bp[3*N]));
            b_reg1.s2 = as_int((half2)(bp[4*N], bp[5*N]));
            b_reg1.s3 = as_int((half2)(bp[6*N], bp[7*N]));
            b_reg1.s4 = as_int((half2)(bp[8*N], bp[9*N]));
            b_reg1.s5 = as_int((half2)(bp[10*N], bp[11*N]));
            b_reg1.s6 = as_int((half2)(bp[12*N], bp[13*N]));
            b_reg1.s7 = as_int((half2)(bp[14*N], bp[15*N]));
        }

        acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0_1, b_reg1, acc0);
        acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1_1, b_reg1, acc1);
        acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2_1, b_reg1, acc2);
        acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3_1, b_reg1, acc3);
    }

    // === Store C using subgroup block writes ===
    // Each subgroup writes 32 rows x 16 cols, one row at a time
    __global half* c_base = C + wg_row * N + sg_col;

    // Use intel_sub_group_block_write_us for 16-wide half stores
    intel_sub_group_block_write_us((__global ushort*)&c_base[0*N], as_ushort(convert_half(acc0.s0)));
    intel_sub_group_block_write_us((__global ushort*)&c_base[1*N], as_ushort(convert_half(acc0.s1)));
    intel_sub_group_block_write_us((__global ushort*)&c_base[2*N], as_ushort(convert_half(acc0.s2)));
    intel_sub_group_block_write_us((__global ushort*)&c_base[3*N], as_ushort(convert_half(acc0.s3)));
    intel_sub_group_block_write_us((__global ushort*)&c_base[4*N], as_ushort(convert_half(acc0.s4)));
    intel_sub_group_block_write_us((__global ushort*)&c_base[5*N], as_ushort(convert_half(acc0.s5)));
    intel_sub_group_block_write_us((__global ushort*)&c_base[6*N], as_ushort(convert_half(acc0.s6)));
    intel_sub_group_block_write_us((__global ushort*)&c_base[7*N], as_ushort(convert_half(acc0.s7)));

    intel_sub_group_block_write_us((__global ushort*)&c_base[8*N], as_ushort(convert_half(acc1.s0)));
    intel_sub_group_block_write_us((__global ushort*)&c_base[9*N], as_ushort(convert_half(acc1.s1)));
    intel_sub_group_block_write_us((__global ushort*)&c_base[10*N], as_ushort(convert_half(acc1.s2)));
    intel_sub_group_block_write_us((__global ushort*)&c_base[11*N], as_ushort(convert_half(acc1.s3)));
    intel_sub_group_block_write_us((__global ushort*)&c_base[12*N], as_ushort(convert_half(acc1.s4)));
    intel_sub_group_block_write_us((__global ushort*)&c_base[13*N], as_ushort(convert_half(acc1.s5)));
    intel_sub_group_block_write_us((__global ushort*)&c_base[14*N], as_ushort(convert_half(acc1.s6)));
    intel_sub_group_block_write_us((__global ushort*)&c_base[15*N], as_ushort(convert_half(acc1.s7)));

    intel_sub_group_block_write_us((__global ushort*)&c_base[16*N], as_ushort(convert_half(acc2.s0)));
    intel_sub_group_block_write_us((__global ushort*)&c_base[17*N], as_ushort(convert_half(acc2.s1)));
    intel_sub_group_block_write_us((__global ushort*)&c_base[18*N], as_ushort(convert_half(acc2.s2)));
    intel_sub_group_block_write_us((__global ushort*)&c_base[19*N], as_ushort(convert_half(acc2.s3)));
    intel_sub_group_block_write_us((__global ushort*)&c_base[20*N], as_ushort(convert_half(acc2.s4)));
    intel_sub_group_block_write_us((__global ushort*)&c_base[21*N], as_ushort(convert_half(acc2.s5)));
    intel_sub_group_block_write_us((__global ushort*)&c_base[22*N], as_ushort(convert_half(acc2.s6)));
    intel_sub_group_block_write_us((__global ushort*)&c_base[23*N], as_ushort(convert_half(acc2.s7)));

    intel_sub_group_block_write_us((__global ushort*)&c_base[24*N], as_ushort(convert_half(acc3.s0)));
    intel_sub_group_block_write_us((__global ushort*)&c_base[25*N], as_ushort(convert_half(acc3.s1)));
    intel_sub_group_block_write_us((__global ushort*)&c_base[26*N], as_ushort(convert_half(acc3.s2)));
    intel_sub_group_block_write_us((__global ushort*)&c_base[27*N], as_ushort(convert_half(acc3.s3)));
    intel_sub_group_block_write_us((__global ushort*)&c_base[28*N], as_ushort(convert_half(acc3.s4)));
    intel_sub_group_block_write_us((__global ushort*)&c_base[29*N], as_ushort(convert_half(acc3.s5)));
    intel_sub_group_block_write_us((__global ushort*)&c_base[30*N], as_ushort(convert_half(acc3.s6)));
    intel_sub_group_block_write_us((__global ushort*)&c_base[31*N], as_ushort(convert_half(acc3.s7)));
}
```

Console output from running this kernel:

Test result on platform Intel Corporation Battlemage G21 [Intel Graphics]:
==== test session starts

task.py::TestMatmulOCL::test_correctness_wrt_pytorch[0] PASSED           [ 25%]
task.py::TestMatmulOCL::test_correctness_wrt_pytorch[1] PASSED           [ 50%]
task.py::TestMatmulOCL::test_correctness_wrt_reference[0] PASSED         [ 75%]
task.py::TestMatmulOCL::test_correctness_wrt_reference[1] PASSED         [100%]

=============================== warnings summary ===============================
task.py::TestMatmulOCL::test_correctness_wrt_pytorch[0]
  /home/openvino-ci-74/miniforge3/envs/kernel_intel/lib/python3.12/site-packages/pyopencl/cache.py:517: CompilerWarning: Non-empty compiler output encountered. Set the environment variable PYOPENCL_COMPILER_OUTPUT=1 to see more.
    _create_built_program_from_source_cached(

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
================== 4 passed, 1 deselected, 1 warning in 0.88s ==================
The kernel compiles and is correct, great job!

## Hardware specification:
Your code will run on the following hardware:
**Intel Battlemage** with specs: Xe-cores: 20, Render Slices: 5, Ray Tracing Units: 20, Intel® Xe Matrix Extensions (Intel® XMX) Engines: 160, Xe Vector Engines: 160, Graphics Clock: 2670, GPU Peak TOPS (Int8): 233, TBP: 190, PCI Express Configurations ‡: PCI Express 4.0 x8, Device ID: 0xE20B, Memory: 12 GB GDDR6, Memory Interface: 192 bit, Memory Bandwidth: 456, Memory Speed: 19, ISA_GPU: Xe2-HPG
Please consider the hardware specifications when improving the code. 

## Task:

**Your objectives**:
1. Analyze the previous versions and their results (why does one achieve better results than the other?).
2. Identify any inefficiencies and bottlenecks.
3. Propose specific improvements or options to take the best of all prior versions, explaining your reasoning step by step.

4. Provide a new kernel that achieves better performance **on the target hardware**. Provide the complete, improved code in a code block.

**Optimization strategies**:

Here are some potential strategies to improve the kernel runtime:
1. Async Memory Operations: Use async_work_group_copy() to overlap computation with memory transfers between global and local memory. Use async_work_group_strided_copy() for non-contiguous data. Wait with wait_group_events().
2. Avoid Bank Conflicts: Local memory is organized into banks (typically 32 banks). Pad shared arrays to avoid stride conflicts, e.g., __local float tile[TILE_SIZE][TILE_SIZE + 1] for transpose operations. Use sequential access patterns within wavefronts.

**Critical Requirements:**

1. The kernel must exactly match the reference implementation's functionality.
2. The code must compile and run properly on the GPU.
3. Do not cache or reuse previous results; ensure the code executes fully on each run.
4. Keep all hyperparameters (e.g., batch size, dimensions) unchanged as specified in the reference implementation.
8. Beware of the critical error "Unexpected kernel lambda size. In such cases removing constexpr specifier aligns the captures between the host compiler and the device compiler"! Do not capture constexpr variables in lambda functions passed to kernel launches as this can lead to different lambda sizes between the host and device compiler.

Additional Guidance:

1. Clearly comment on any performance optimizations you implement.
2. If you change the kernel structure, explain why.
3. Anticipate and address possible runtime or compilation errors.

Please structure your response as follows:

1. Analysis:
    * Summarize the issues found in the previous kernel and log.
    * Explain your proposed changes and optimizations.
2. Improved OCL code:
    * Provide the complete, improved OCL code in a code block:
```OCL
Your code here
```


## Required Optimizations

Apply the following optimization techniques in your implementation:

- **Register Blocking**: Each work-item computes a THREAD_M×THREAD_N output block in private register arrays. Use `#pragma unroll` on inner loops. Combine with SLM tiling for multi-level memory hierarchy optimization.
- **Blocked/Tiled Algorithms**: Process input in blocks to bound peak memory. Trade recomputation for memory savings (e.g., Flash-Attention style). Maintain running accumulators across blocks with proper rescaling.
- **Hierarchical Parallelism**: Structure work at three levels - work-groups (tile assignment), sub-groups (sub-tile processing), work-items (register tile). Use sub-group shuffles to share data within sub-group without SLM.