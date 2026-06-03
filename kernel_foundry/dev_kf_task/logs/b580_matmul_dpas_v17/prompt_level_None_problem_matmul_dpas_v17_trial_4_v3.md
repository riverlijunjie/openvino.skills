

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

### Version 1 (Result: Correctness score: 5 / 5 (compiles and correct), runtime in ms: 0.831):
```OCL
// GEMM: C[M,N] = A[M,K] x B[K,N], row-major, FP16 in/out, FP32 accumulation
// Architecture: 64 WIs (4 SGs of 16), TILE 32x64x32, A in SLM (double-buffered), B from global
// Launch: GWS = (N/64 * 64, M/32), LWS = (64, 1)
// Each WG computes a 32(M) x 64(N) output tile
// 4 subgroups, each handles 32 rows x 16 cols
// K-loop steps by 32, double-buffered A in SLM
// Optimizations: SLM stride=34 for bank conflict reduction, interleaved B/DPAS,
//   half16 vector loads for A global->SLM, separate last iteration to remove branch

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void matmul(
    __global const half* restrict A,
    __global const half* restrict B,
    __global half* restrict C,
    const int M,
    const int K,
    const int N)
{
    const int wg_n = get_group_id(0);
    const int wg_m = get_group_id(1);

    const int lid = get_local_id(0);
    const int sg_id = lid / 16;
    const int sg_lid = get_sub_group_local_id();

    const int baseM = wg_m * 32;
    const int baseN = wg_n * 64 + sg_id * 16;

    // SLM stride=34 to reduce bank conflicts (34*2=68 bytes per row, not power-of-2)
    #define A_SLM_STRIDE 34
    #define A_SLM_SIZE (32 * A_SLM_STRIDE)
    __local half slm_A[2 * A_SLM_SIZE];

    // Accumulators: 32 rows x 16 cols = 4 x float8
    float8 acc0 = (float8)(0.0f);
    float8 acc1 = (float8)(0.0f);
    float8 acc2 = (float8)(0.0f);
    float8 acc3 = (float8)(0.0f);

    // Cooperative A load: 64 WIs load 32x32 = 1024 halfs = 16 halfs/WI
    const int a_load_row = lid / 2;
    const int a_load_col = (lid & 1) * 16;
    const int a_gm_row = baseM + a_load_row;

    // Preload first A tile (k=0..31) into SLM buffer 0
    {
        __global const half* a_src = A + a_gm_row * K + a_load_col;
        __local half* a_dst = slm_A + a_load_row * A_SLM_STRIDE + a_load_col;
        *(__local half16*)a_dst = *(__global const half16*)a_src;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const int k_iters = K / 32;  // 64
    int cur_buf = 0;

    // Main loop: all iterations except the last (no next A load needed for last)
    for (int ki = 0; ki < k_iters - 1; ki++) {
        const int k_base = ki * 32;
        const int next_buf = 1 - cur_buf;

        __local const half* a_slm = slm_A + cur_buf * A_SLM_SIZE;

        // === First k16 step: k_base..k_base+15 ===

        // Load B for first k16
        int8 b0;
        {
            __global const half* b_col = B + k_base * N + baseN + sg_lid;
            ushort bv0  = as_ushort(b_col[0 * N]);
            ushort bv1  = as_ushort(b_col[1 * N]);
            ushort bv2  = as_ushort(b_col[2 * N]);
            ushort bv3  = as_ushort(b_col[3 * N]);
            ushort bv4  = as_ushort(b_col[4 * N]);
            ushort bv5  = as_ushort(b_col[5 * N]);
            ushort bv6  = as_ushort(b_col[6 * N]);
            ushort bv7  = as_ushort(b_col[7 * N]);
            ushort bv8  = as_ushort(b_col[8 * N]);
            ushort bv9  = as_ushort(b_col[9 * N]);
            ushort bv10 = as_ushort(b_col[10 * N]);
            ushort bv11 = as_ushort(b_col[11 * N]);
            ushort bv12 = as_ushort(b_col[12 * N]);
            ushort bv13 = as_ushort(b_col[13 * N]);
            ushort bv14 = as_ushort(b_col[14 * N]);
            ushort bv15 = as_ushort(b_col[15 * N]);

            b0.s0 = as_int((ushort2)(bv0, bv1));
            b0.s1 = as_int((ushort2)(bv2, bv3));
            b0.s2 = as_int((ushort2)(bv4, bv5));
            b0.s3 = as_int((ushort2)(bv6, bv7));
            b0.s4 = as_int((ushort2)(bv8, bv9));
            b0.s5 = as_int((ushort2)(bv10, bv11));
            b0.s6 = as_int((ushort2)(bv12, bv13));
            b0.s7 = as_int((ushort2)(bv14, bv15));
        }

        // Load A from SLM for first k16
        short8 a00, a01, a02, a03;
        {
            __local const half* a_k0 = a_slm + sg_lid;
            a00.s0 = as_short(a_k0[0 * A_SLM_STRIDE]);
            a00.s1 = as_short(a_k0[1 * A_SLM_STRIDE]);
            a00.s2 = as_short(a_k0[2 * A_SLM_STRIDE]);
            a00.s3 = as_short(a_k0[3 * A_SLM_STRIDE]);
            a00.s4 = as_short(a_k0[4 * A_SLM_STRIDE]);
            a00.s5 = as_short(a_k0[5 * A_SLM_STRIDE]);
            a00.s6 = as_short(a_k0[6 * A_SLM_STRIDE]);
            a00.s7 = as_short(a_k0[7 * A_SLM_STRIDE]);

            a01.s0 = as_short(a_k0[8 * A_SLM_STRIDE]);
            a01.s1 = as_short(a_k0[9 * A_SLM_STRIDE]);
            a01.s2 = as_short(a_k0[10 * A_SLM_STRIDE]);
            a01.s3 = as_short(a_k0[11 * A_SLM_STRIDE]);
            a01.s4 = as_short(a_k0[12 * A_SLM_STRIDE]);
            a01.s5 = as_short(a_k0[13 * A_SLM_STRIDE]);
            a01.s6 = as_short(a_k0[14 * A_SLM_STRIDE]);
            a01.s7 = as_short(a_k0[15 * A_SLM_STRIDE]);

            a02.s0 = as_short(a_k0[16 * A_SLM_STRIDE]);
            a02.s1 = as_short(a_k0[17 * A_SLM_STRIDE]);
            a02.s2 = as_short(a_k0[18 * A_SLM_STRIDE]);
            a02.s3 = as_short(a_k0[19 * A_SLM_STRIDE]);
            a02.s4 = as_short(a_k0[20 * A_SLM_STRIDE]);
            a02.s5 = as_short(a_k0[21 * A_SLM_STRIDE]);
            a02.s6 = as_short(a_k0[22 * A_SLM_STRIDE]);
            a02.s7 = as_short(a_k0[23 * A_SLM_STRIDE]);

            a03.s0 = as_short(a_k0[24 * A_SLM_STRIDE]);
            a03.s1 = as_short(a_k0[25 * A_SLM_STRIDE]);
            a03.s2 = as_short(a_k0[26 * A_SLM_STRIDE]);
            a03.s3 = as_short(a_k0[27 * A_SLM_STRIDE]);
            a03.s4 = as_short(a_k0[28 * A_SLM_STRIDE]);
            a03.s5 = as_short(a_k0[29 * A_SLM_STRIDE]);
            a03.s6 = as_short(a_k0[30 * A_SLM_STRIDE]);
            a03.s7 = as_short(a_k0[31 * A_SLM_STRIDE]);
        }

        // DPAS first k16 - interleave with next A tile load
        acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a00, b0, acc0);
        acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a01, b0, acc1);

        // Interleave: load next A tile into other SLM buffer while DPAS executes
        __global const half* a_next_src = A + a_gm_row * K + (k_base + 32) + a_load_col;
        __local half* a_next_dst = slm_A + next_buf * A_SLM_SIZE + a_load_row * A_SLM_STRIDE + a_load_col;
        *(__local half16*)a_next_dst = *(__global const half16*)a_next_src;

        acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a02, b0, acc2);
        acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a03, b0, acc3);

        // === Second k16 step: k_base+16..k_base+31 ===

        // Load B for second k16
        int8 b1;
        {
            __global const half* b_col = B + (k_base + 16) * N + baseN + sg_lid;
            ushort bv0  = as_ushort(b_col[0 * N]);
            ushort bv1  = as_ushort(b_col[1 * N]);
            ushort bv2  = as_ushort(b_col[2 * N]);
            ushort bv3  = as_ushort(b_col[3 * N]);
            ushort bv4  = as_ushort(b_col[4 * N]);
            ushort bv5  = as_ushort(b_col[5 * N]);
            ushort bv6  = as_ushort(b_col[6 * N]);
            ushort bv7  = as_ushort(b_col[7 * N]);
            ushort bv8  = as_ushort(b_col[8 * N]);
            ushort bv9  = as_ushort(b_col[9 * N]);
            ushort bv10 = as_ushort(b_col[10 * N]);
            ushort bv11 = as_ushort(b_col[11 * N]);
            ushort bv12 = as_ushort(b_col[12 * N]);
            ushort bv13 = as_ushort(b_col[13 * N]);
            ushort bv14 = as_ushort(b_col[14 * N]);
            ushort bv15 = as_ushort(b_col[15 * N]);

            b1.s0 = as_int((ushort2)(bv0, bv1));
            b1.s1 = as_int((ushort2)(bv2, bv3));
            b1.s2 = as_int((ushort2)(bv4, bv5));
            b1.s3 = as_int((ushort2)(bv6, bv7));
            b1.s4 = as_int((ushort2)(bv8, bv9));
            b1.s5 = as_int((ushort2)(bv10, bv11));
            b1.s6 = as_int((ushort2)(bv12, bv13));
            b1.s7 = as_int((ushort2)(bv14, bv15));
        }

        // Load A from SLM for second k16
        short8 a10, a11, a12, a13;
        {
            __local const half* a_k1 = a_slm + 16 + sg_lid;
            a10.s0 = as_short(a_k1[0 * A_SLM_STRIDE]);
            a10.s1 = as_short(a_k1[1 * A_SLM_STRIDE]);
            a10.s2 = as_short(a_k1[2 * A_SLM_STRIDE]);
            a10.s3 = as_short(a_k1[3 * A_SLM_STRIDE]);
            a10.s4 = as_short(a_k1[4 * A_SLM_STRIDE]);
            a10.s5 = as_short(a_k1[5 * A_SLM_STRIDE]);
            a10.s6 = as_short(a_k1[6 * A_SLM_STRIDE]);
            a10.s7 = as_short(a_k1[7 * A_SLM_STRIDE]);

            a11.s0 = as_short(a_k1[8 * A_SLM_STRIDE]);
            a11.s1 = as_short(a_k1[9 * A_SLM_STRIDE]);
            a11.s2 = as_short(a_k1[10 * A_SLM_STRIDE]);
            a11.s3 = as_short(a_k1[11 * A_SLM_STRIDE]);
            a11.s4 = as_short(a_k1[12 * A_SLM_STRIDE]);
            a11.s5 = as_short(a_k1[13 * A_SLM_STRIDE]);
            a11.s6 = as_short(a_k1[14 * A_SLM_STRIDE]);
            a11.s7 = as_short(a_k1[15 * A_SLM_STRIDE]);

            a12.s0 = as_short(a_k1[16 * A_SLM_STRIDE]);
            a12.s1 = as_short(a_k1[17 * A_SLM_STRIDE]);
            a12.s2 = as_short(a_k1[18 * A_SLM_STRIDE]);
            a12.s3 = as_short(a_k1[19 * A_SLM_STRIDE]);
            a12.s4 = as_short(a_k1[20 * A_SLM_STRIDE]);
            a12.s5 = as_short(a_k1[21 * A_SLM_STRIDE]);
            a12.s6 = as_short(a_k1[22 * A_SLM_STRIDE]);
            a12.s7 = as_short(a_k1[23 * A_SLM_STRIDE]);

            a13.s0 = as_short(a_k1[24 * A_SLM_STRIDE]);
            a13.s1 = as_short(a_k1[25 * A_SLM_STRIDE]);
            a13.s2 = as_short(a_k1[26 * A_SLM_STRIDE]);
            a13.s3 = as_short(a_k1[27 * A_SLM_STRIDE]);
            a13.s4 = as_short(a_k1[28 * A_SLM_STRIDE]);
            a13.s5 = as_short(a_k1[29 * A_SLM_STRIDE]);
            a13.s6 = as_short(a_k1[30 * A_SLM_STRIDE]);
            a13.s7 = as_short(a_k1[31 * A_SLM_STRIDE]);
        }

        // DPAS second k16
        acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a10, b1, acc0);
        acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a11, b1, acc1);
        acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a12, b1, acc2);
        acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a13, b1, acc3);

        cur_buf = next_buf;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // === Last iteration (ki = k_iters - 1): no next A load needed ===
    {
        const int k_base = (k_iters - 1) * 32;
        __local const half* a_slm = slm_A + cur_buf * A_SLM_SIZE;

        // First k16
        int8 b0;
        {
            __global const half* b_col = B + k_base * N + baseN + sg_lid;
            ushort bv0  = as_ushort(b_col[0 * N]);
            ushort bv1  = as_ushort(b_col[1 * N]);
            ushort bv2  = as_ushort(b_col[2 * N]);
            ushort bv3  = as_ushort(b_col[3 * N]);
            ushort bv4  = as_ushort(b_col[4 * N]);
            ushort bv5  = as_ushort(b_col[5 * N]);
            ushort bv6  = as_ushort(b_col[6 * N]);
            ushort bv7  = as_ushort(b_col[7 * N]);
            ushort bv8  = as_ushort(b_col[8 * N]);
            ushort bv9  = as_ushort(b_col[9 * N]);
            ushort bv10 = as_ushort(b_col[10 * N]);
            ushort bv11 = as_ushort(b_col[11 * N]);
            ushort bv12 = as_ushort(b_col[12 * N]);
            ushort bv13 = as_ushort(b_col[13 * N]);
            ushort bv14 = as_ushort(b_col[14 * N]);
            ushort bv15 = as_ushort(b_col[15 * N]);

            b0.s0 = as_int((ushort2)(bv0, bv1));
            b0.s1 = as_int((ushort2)(bv2, bv3));
            b0.s2 = as_int((ushort2)(bv4, bv5));
            b0.s3 = as_int((ushort2)(bv6, bv7));
            b0.s4 = as_int((ushort2)(bv8, bv9));
            b0.s5 = as_int((ushort2)(bv10, bv11));
            b0.s6 = as_int((ushort2)(bv12, bv13));
            b0.s7 = as_int((ushort2)(bv14, bv15));
        }

        short8 a00, a01, a02, a03;
        {
            __local const half* a_k0 = a_slm + sg_lid;
            a00.s0 = as_short(a_k0[0 * A_SLM_STRIDE]);
            a00.s1 = as_short(a_k0[1 * A_SLM_STRIDE]);
            a00.s2 = as_short(a_k0[2 * A_SLM_STRIDE]);
            a00.s3 = as_short(a_k0[3 * A_SLM_STRIDE]);
            a00.s4 = as_short(a_k0[4 * A_SLM_STRIDE]);
            a00.s5 = as_short(a_k0[5 * A_SLM_STRIDE]);
            a00.s6 = as_short(a_k0[6 * A_SLM_STRIDE]);
            a00.s7 = as_short(a_k0[7 * A_SLM_STRIDE]);

            a01.s0 = as_short(a_k0[8 * A_SLM_STRIDE]);
            a01.s1 = as_short(a_k0[9 * A_SLM_STRIDE]);
            a01.s2 = as_short(a_k0[10 * A_SLM_STRIDE]);
            a01.s3 = as_short(a_k0[11 * A_SLM_STRIDE]);
            a01.s4 = as_short(a_k0[12 * A_SLM_STRIDE]);
            a01.s5 = as_short(a_k0[13 * A_SLM_STRIDE]);
            a01.s6 = as_short(a_k0[14 * A_SLM_STRIDE]);
            a01.s7 = as_short(a_k0[15 * A_SLM_STRIDE]);

            a02.s0 = as_short(a_k0[16 * A_SLM_STRIDE]);
            a02.s1 = as_short(a_k0[17 * A_SLM_STRIDE]);
            a02.s2 = as_short(a_k0[18 * A_SLM_STRIDE]);
            a02.s3 = as_short(a_k0[19 * A_SLM_STRIDE]);
            a02.s4 = as_short(a_k0[20 * A_SLM_STRIDE]);
            a02.s5 = as_short(a_k0[21 * A_SLM_STRIDE]);
            a02.s6 = as_short(a_k0[22 * A_SLM_STRIDE]);
            a02.s7 = as_short(a_k0[23 * A_SLM_STRIDE]);

            a03.s0 = as_short(a_k0[24 * A_SLM_STRIDE]);
            a03.s1 = as_short(a_k0[25 * A_SLM_STRIDE]);
            a03.s2 = as_short(a_k0[26 * A_SLM_STRIDE]);
            a03.s3 = as_short(a_k0[27 * A_SLM_STRIDE]);
            a03.s4 = as_short(a_k0[28 * A_SLM_STRIDE]);
            a03.s5 = as_short(a_k0[29 * A_SLM_STRIDE]);
            a03.s6 = as_short(a_k0[30 * A_SLM_STRIDE]);
            a03.s7 = as_short(a_k0[31 * A_SLM_STRIDE]);
        }

        acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a00, b0, acc0);
        acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a01, b0, acc1);
        acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a02, b0, acc2);
        acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a03, b0, acc3);

        // Second k16
        int8 b1;
        {
            __global const half* b_col = B + (k_base + 16) * N + baseN + sg_lid;
            ushort bv0  = as_ushort(b_col[0 * N]);
            ushort bv1  = as_ushort(b_col[1 * N]);
            ushort bv2  = as_ushort(b_col[2 * N]);
            ushort bv3  = as_ushort(b_col[3 * N]);
            ushort bv4  = as_ushort(b_col[4 * N]);
            ushort bv5  = as_ushort(b_col[5 * N]);
            ushort bv6  = as_ushort(b_col[6 * N]);
            ushort bv7  = as_ushort(b_col[7 * N]);
            ushort bv8  = as_ushort(b_col[8 * N]);
            ushort bv9  = as_ushort(b_col[9 * N]);
            ushort bv10 = as_ushort(b_col[10 * N]);
            ushort bv11 = as_ushort(b_col[11 * N]);
            ushort bv12 = as_ushort(b_col[12 * N]);
            ushort bv13 = as_ushort(b_col[13 * N]);
            ushort bv14 = as_ushort(b_col[14 * N]);
            ushort bv15 = as_ushort(b_col[15 * N]);

            b1.s0 = as_int((ushort2)(bv0, bv1));
            b1.s1 = as_int((ushort2)(bv2, bv3));
            b1.s2 = as_int((ushort2)(bv4, bv5));
            b1.s3 = as_int((ushort2)(bv6, bv7));
            b1.s4 = as_int((ushort2)(bv8, bv9));
            b1.s5 = as_int((ushort2)(bv10, bv11));
            b1.s6 = as_int((ushort2)(bv12, bv13));
            b1.s7 = as_int((ushort2)(bv14, bv15));
        }

        short8 a10, a11, a12, a13;
        {
            __local const half* a_k1 = a_slm + 16 + sg_lid;
            a10.s0 = as_short(a_k1[0 * A_SLM_STRIDE]);
            a10.s1 = as_short(a_k1[1 * A_SLM_STRIDE]);
            a10.s2 = as_short(a_k1[2 * A_SLM_STRIDE]);
            a10.s3 = as_short(a_k1[3 * A_SLM_STRIDE]);
            a10.s4 = as_short(a_k1[4 * A_SLM_STRIDE]);
            a10.s5 = as_short(a_k1[5 * A_SLM_STRIDE]);
            a10.s6 = as_short(a_k1[6 * A_SLM_STRIDE]);
            a10.s7 = as_short(a_k1[7 * A_SLM_STRIDE]);

            a11.s0 = as_short(a_k1[8 * A_SLM_STRIDE]);
            a11.s1 = as_short(a_k1[9 * A_SLM_STRIDE]);
            a11.s2 = as_short(a_k1[10 * A_SLM_STRIDE]);
            a11.s3 = as_short(a_k1[11 * A_SLM_STRIDE]);
            a11.s4 = as_short(a_k1[12 * A_SLM_STRIDE]);
            a11.s5 = as_short(a_k1[13 * A_SLM_STRIDE]);
            a11.s6 = as_short(a_k1[14 * A_SLM_STRIDE]);
            a11.s7 = as_short(a_k1[15 * A_SLM_STRIDE]);

            a12.s0 = as_short(a_k1[16 * A_SLM_STRIDE]);
            a12.s1 = as_short(a_k1[17 * A_SLM_STRIDE]);
            a12.s2 = as_short(a_k1[18 * A_SLM_STRIDE]);
            a12.s3 = as_short(a_k1[19 * A_SLM_STRIDE]);
            a12.s4 = as_short(a_k1[20 * A_SLM_STRIDE]);
            a12.s5 = as_short(a_k1[21 * A_SLM_STRIDE]);
            a12.s6 = as_short(a_k1[22 * A_SLM_STRIDE]);
            a12.s7 = as_short(a_k1[23 * A_SLM_STRIDE]);

            a13.s0 = as_short(a_k1[24 * A_SLM_STRIDE]);
            a13.s1 = as_short(a_k1[25 * A_SLM_STRIDE]);
            a13.s2 = as_short(a_k1[26 * A_SLM_STRIDE]);
            a13.s3 = as_short(a_k1[27 * A_SLM_STRIDE]);
            a13.s4 = as_short(a_k1[28 * A_SLM_STRIDE]);
            a13.s5 = as_short(a_k1[29 * A_SLM_STRIDE]);
            a13.s6 = as_short(a_k1[30 * A_SLM_STRIDE]);
            a13.s7 = as_short(a_k1[31 * A_SLM_STRIDE]);
        }

        acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a10, b1, acc0);
        acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a11, b1, acc1);
        acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a12, b1, acc2);
        acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a13, b1, acc3);
    }

    // Store C: each subgroup writes 32 rows x 16 cols
    __global half* c_out = C + baseM * N + baseN + sg_lid;

    #pragma unroll
    for (int r = 0; r < 8; r++) {
        c_out[(0 + r) * N] = convert_half(((float*)&acc0)[r]);
    }
    #pragma unroll
    for (int r = 0; r < 8; r++) {
        c_out[(8 + r) * N] = convert_half(((float*)&acc1)[r]);
    }
    #pragma unroll
    for (int r = 0; r < 8; r++) {
        c_out[(16 + r) * N] = convert_half(((float*)&acc2)[r]);
    }
    #pragma unroll
    for (int r = 0; r < 8; r++) {
        c_out[(24 + r) * N] = convert_half(((float*)&acc3)[r]);
    }

    #undef A_SLM_STRIDE
    #undef A_SLM_SIZE
}
```

### Version 2 (Result: Correctness score: 5 / 5 (compiles and correct), runtime in ms: 0.832):
```OCL
// GEMM: C[M,N] = A[M,K] x B[K,N], row-major, FP16 in/out, FP32 accumulation
// Architecture: 64 WIs (4 SGs of 16), TILE 32x64x32, A in SLM (double-buffered), B from global
// Launch: GWS = (N/64 * 64, M/32), LWS = (64, 1)
// Each WG computes a 32(M) x 64(N) output tile
// 4 subgroups, each handles 32 rows x 16 cols
// K-loop steps by 32, double-buffered A in SLM
// Optimizations: vectorized A→SLM copy, interleaved B load + DPAS, no last-iter branch

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void matmul(
    __global const half* restrict A,
    __global const half* restrict B,
    __global half* restrict C,
    const int M,
    const int K,
    const int N)
{
    const int wg_n = get_group_id(0);
    const int wg_m = get_group_id(1);

    const int lid = get_local_id(0);
    const int sg_id = lid / 16;
    const int sg_lid = get_sub_group_local_id();

    const int baseM = wg_m * 32;
    const int baseN = wg_n * 64 + sg_id * 16;

    // SLM for A: double-buffered, 32 rows x 32 cols
    // Use stride=36 to reduce bank conflicts (36*2=72 bytes between rows)
    #define A_SLM_STRIDE 36
    #define A_SLM_SIZE (32 * A_SLM_STRIDE)
    __local half slm_A[2 * A_SLM_SIZE];

    // Accumulators: 32 rows x 16 cols = 4 x float8
    float8 acc0 = (float8)(0.0f);
    float8 acc1 = (float8)(0.0f);
    float8 acc2 = (float8)(0.0f);
    float8 acc3 = (float8)(0.0f);

    // Cooperative A load: 64 WIs load 32x32 = 1024 halfs = 16 halfs/WI
    const int a_load_row = lid / 2;
    const int a_load_col = (lid & 1) * 16;
    const int a_gm_row = baseM + a_load_row;

    // Preload first A tile (k=0..31) into SLM buffer 0
    {
        __global const half* a_src = A + a_gm_row * K + a_load_col;
        __local half* a_dst = slm_A + a_load_row * A_SLM_STRIDE + a_load_col;
        *(__local half16*)a_dst = *(__global const half16*)a_src;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // K=2048, step=32 -> 64 iterations
    const int k_iters = K / 32;  // 64
    int cur_buf = 0;

    // Main loop: all iterations except the last (no next A load on last)
    for (int ki = 0; ki < k_iters - 1; ki++) {
        const int k_base = ki * 32;
        const int next_buf = 1 - cur_buf;

        __local const half* a_slm = slm_A + cur_buf * A_SLM_SIZE;

        // === First k16 step: k_base..k_base+15 ===

        // Load B for first k16 - use strided scalar loads (B is row-major, stride=N)
        int8 b0;
        {
            __global const half* b_ptr = B + k_base * N + baseN + sg_lid;
            ushort bv0  = as_ushort(b_ptr[0 * N]);
            ushort bv1  = as_ushort(b_ptr[1 * N]);
            ushort bv2  = as_ushort(b_ptr[2 * N]);
            ushort bv3  = as_ushort(b_ptr[3 * N]);
            ushort bv4  = as_ushort(b_ptr[4 * N]);
            ushort bv5  = as_ushort(b_ptr[5 * N]);
            ushort bv6  = as_ushort(b_ptr[6 * N]);
            ushort bv7  = as_ushort(b_ptr[7 * N]);
            ushort bv8  = as_ushort(b_ptr[8 * N]);
            ushort bv9  = as_ushort(b_ptr[9 * N]);
            ushort bv10 = as_ushort(b_ptr[10 * N]);
            ushort bv11 = as_ushort(b_ptr[11 * N]);
            ushort bv12 = as_ushort(b_ptr[12 * N]);
            ushort bv13 = as_ushort(b_ptr[13 * N]);
            ushort bv14 = as_ushort(b_ptr[14 * N]);
            ushort bv15 = as_ushort(b_ptr[15 * N]);
            b0.s0 = as_int((ushort2)(bv0, bv1));
            b0.s1 = as_int((ushort2)(bv2, bv3));
            b0.s2 = as_int((ushort2)(bv4, bv5));
            b0.s3 = as_int((ushort2)(bv6, bv7));
            b0.s4 = as_int((ushort2)(bv8, bv9));
            b0.s5 = as_int((ushort2)(bv10, bv11));
            b0.s6 = as_int((ushort2)(bv12, bv13));
            b0.s7 = as_int((ushort2)(bv14, bv15));
        }

        // Load A from SLM for first k16
        short8 a00, a01, a02, a03;
        {
            __local const half* a_k0 = a_slm + sg_lid;
            a00.s0 = as_short(a_k0[0 * A_SLM_STRIDE]);
            a00.s1 = as_short(a_k0[1 * A_SLM_STRIDE]);
            a00.s2 = as_short(a_k0[2 * A_SLM_STRIDE]);
            a00.s3 = as_short(a_k0[3 * A_SLM_STRIDE]);
            a00.s4 = as_short(a_k0[4 * A_SLM_STRIDE]);
            a00.s5 = as_short(a_k0[5 * A_SLM_STRIDE]);
            a00.s6 = as_short(a_k0[6 * A_SLM_STRIDE]);
            a00.s7 = as_short(a_k0[7 * A_SLM_STRIDE]);

            a01.s0 = as_short(a_k0[8 * A_SLM_STRIDE]);
            a01.s1 = as_short(a_k0[9 * A_SLM_STRIDE]);
            a01.s2 = as_short(a_k0[10 * A_SLM_STRIDE]);
            a01.s3 = as_short(a_k0[11 * A_SLM_STRIDE]);
            a01.s4 = as_short(a_k0[12 * A_SLM_STRIDE]);
            a01.s5 = as_short(a_k0[13 * A_SLM_STRIDE]);
            a01.s6 = as_short(a_k0[14 * A_SLM_STRIDE]);
            a01.s7 = as_short(a_k0[15 * A_SLM_STRIDE]);

            a02.s0 = as_short(a_k0[16 * A_SLM_STRIDE]);
            a02.s1 = as_short(a_k0[17 * A_SLM_STRIDE]);
            a02.s2 = as_short(a_k0[18 * A_SLM_STRIDE]);
            a02.s3 = as_short(a_k0[19 * A_SLM_STRIDE]);
            a02.s4 = as_short(a_k0[20 * A_SLM_STRIDE]);
            a02.s5 = as_short(a_k0[21 * A_SLM_STRIDE]);
            a02.s6 = as_short(a_k0[22 * A_SLM_STRIDE]);
            a02.s7 = as_short(a_k0[23 * A_SLM_STRIDE]);

            a03.s0 = as_short(a_k0[24 * A_SLM_STRIDE]);
            a03.s1 = as_short(a_k0[25 * A_SLM_STRIDE]);
            a03.s2 = as_short(a_k0[26 * A_SLM_STRIDE]);
            a03.s3 = as_short(a_k0[27 * A_SLM_STRIDE]);
            a03.s4 = as_short(a_k0[28 * A_SLM_STRIDE]);
            a03.s5 = as_short(a_k0[29 * A_SLM_STRIDE]);
            a03.s6 = as_short(a_k0[30 * A_SLM_STRIDE]);
            a03.s7 = as_short(a_k0[31 * A_SLM_STRIDE]);
        }

        // DPAS first k16 - interleave with next A tile load
        acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a00, b0, acc0);
        acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a01, b0, acc1);

        // Interleave: load next A tile into alternate SLM buffer while DPAS executes
        __global const half* a_next_src = A + a_gm_row * K + (k_base + 32) + a_load_col;
        __local half* a_next_dst = slm_A + next_buf * A_SLM_SIZE + a_load_row * A_SLM_STRIDE + a_load_col;
        *(__local half16*)a_next_dst = *(__global const half16*)a_next_src;

        acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a02, b0, acc2);
        acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a03, b0, acc3);

        // === Second k16 step: k_base+16..k_base+31 ===

        // Load B for second k16
        int8 b1;
        {
            __global const half* b_ptr = B + (k_base + 16) * N + baseN + sg_lid;
            ushort bv0  = as_ushort(b_ptr[0 * N]);
            ushort bv1  = as_ushort(b_ptr[1 * N]);
            ushort bv2  = as_ushort(b_ptr[2 * N]);
            ushort bv3  = as_ushort(b_ptr[3 * N]);
            ushort bv4  = as_ushort(b_ptr[4 * N]);
            ushort bv5  = as_ushort(b_ptr[5 * N]);
            ushort bv6  = as_ushort(b_ptr[6 * N]);
            ushort bv7  = as_ushort(b_ptr[7 * N]);
            ushort bv8  = as_ushort(b_ptr[8 * N]);
            ushort bv9  = as_ushort(b_ptr[9 * N]);
            ushort bv10 = as_ushort(b_ptr[10 * N]);
            ushort bv11 = as_ushort(b_ptr[11 * N]);
            ushort bv12 = as_ushort(b_ptr[12 * N]);
            ushort bv13 = as_ushort(b_ptr[13 * N]);
            ushort bv14 = as_ushort(b_ptr[14 * N]);
            ushort bv15 = as_ushort(b_ptr[15 * N]);
            b1.s0 = as_int((ushort2)(bv0, bv1));
            b1.s1 = as_int((ushort2)(bv2, bv3));
            b1.s2 = as_int((ushort2)(bv4, bv5));
            b1.s3 = as_int((ushort2)(bv6, bv7));
            b1.s4 = as_int((ushort2)(bv8, bv9));
            b1.s5 = as_int((ushort2)(bv10, bv11));
            b1.s6 = as_int((ushort2)(bv12, bv13));
            b1.s7 = as_int((ushort2)(bv14, bv15));
        }

        // Load A from SLM for second k16
        short8 a10, a11, a12, a13;
        {
            __local const half* a_k1 = a_slm + 16 + sg_lid;
            a10.s0 = as_short(a_k1[0 * A_SLM_STRIDE]);
            a10.s1 = as_short(a_k1[1 * A_SLM_STRIDE]);
            a10.s2 = as_short(a_k1[2 * A_SLM_STRIDE]);
            a10.s3 = as_short(a_k1[3 * A_SLM_STRIDE]);
            a10.s4 = as_short(a_k1[4 * A_SLM_STRIDE]);
            a10.s5 = as_short(a_k1[5 * A_SLM_STRIDE]);
            a10.s6 = as_short(a_k1[6 * A_SLM_STRIDE]);
            a10.s7 = as_short(a_k1[7 * A_SLM_STRIDE]);

            a11.s0 = as_short(a_k1[8 * A_SLM_STRIDE]);
            a11.s1 = as_short(a_k1[9 * A_SLM_STRIDE]);
            a11.s2 = as_short(a_k1[10 * A_SLM_STRIDE]);
            a11.s3 = as_short(a_k1[11 * A_SLM_STRIDE]);
            a11.s4 = as_short(a_k1[12 * A_SLM_STRIDE]);
            a11.s5 = as_short(a_k1[13 * A_SLM_STRIDE]);
            a11.s6 = as_short(a_k1[14 * A_SLM_STRIDE]);
            a11.s7 = as_short(a_k1[15 * A_SLM_STRIDE]);

            a12.s0 = as_short(a_k1[16 * A_SLM_STRIDE]);
            a12.s1 = as_short(a_k1[17 * A_SLM_STRIDE]);
            a12.s2 = as_short(a_k1[18 * A_SLM_STRIDE]);
            a12.s3 = as_short(a_k1[19 * A_SLM_STRIDE]);
            a12.s4 = as_short(a_k1[20 * A_SLM_STRIDE]);
            a12.s5 = as_short(a_k1[21 * A_SLM_STRIDE]);
            a12.s6 = as_short(a_k1[22 * A_SLM_STRIDE]);
            a12.s7 = as_short(a_k1[23 * A_SLM_STRIDE]);

            a13.s0 = as_short(a_k1[24 * A_SLM_STRIDE]);
            a13.s1 = as_short(a_k1[25 * A_SLM_STRIDE]);
            a13.s2 = as_short(a_k1[26 * A_SLM_STRIDE]);
            a13.s3 = as_short(a_k1[27 * A_SLM_STRIDE]);
            a13.s4 = as_short(a_k1[28 * A_SLM_STRIDE]);
            a13.s5 = as_short(a_k1[29 * A_SLM_STRIDE]);
            a13.s6 = as_short(a_k1[30 * A_SLM_STRIDE]);
            a13.s7 = as_short(a_k1[31 * A_SLM_STRIDE]);
        }

        // DPAS second k16
        acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a10, b1, acc0);
        acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a11, b1, acc1);
        acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a12, b1, acc2);
        acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a13, b1, acc3);

        cur_buf = next_buf;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // === Last iteration (ki = k_iters - 1): no next A load ===
    {
        const int k_base = (k_iters - 1) * 32;
        __local const half* a_slm = slm_A + cur_buf * A_SLM_SIZE;

        // First k16
        int8 b0;
        {
            __global const half* b_ptr = B + k_base * N + baseN + sg_lid;
            ushort bv0  = as_ushort(b_ptr[0 * N]);
            ushort bv1  = as_ushort(b_ptr[1 * N]);
            ushort bv2  = as_ushort(b_ptr[2 * N]);
            ushort bv3  = as_ushort(b_ptr[3 * N]);
            ushort bv4  = as_ushort(b_ptr[4 * N]);
            ushort bv5  = as_ushort(b_ptr[5 * N]);
            ushort bv6  = as_ushort(b_ptr[6 * N]);
            ushort bv7  = as_ushort(b_ptr[7 * N]);
            ushort bv8  = as_ushort(b_ptr[8 * N]);
            ushort bv9  = as_ushort(b_ptr[9 * N]);
            ushort bv10 = as_ushort(b_ptr[10 * N]);
            ushort bv11 = as_ushort(b_ptr[11 * N]);
            ushort bv12 = as_ushort(b_ptr[12 * N]);
            ushort bv13 = as_ushort(b_ptr[13 * N]);
            ushort bv14 = as_ushort(b_ptr[14 * N]);
            ushort bv15 = as_ushort(b_ptr[15 * N]);
            b0.s0 = as_int((ushort2)(bv0, bv1));
            b0.s1 = as_int((ushort2)(bv2, bv3));
            b0.s2 = as_int((ushort2)(bv4, bv5));
            b0.s3 = as_int((ushort2)(bv6, bv7));
            b0.s4 = as_int((ushort2)(bv8, bv9));
            b0.s5 = as_int((ushort2)(bv10, bv11));
            b0.s6 = as_int((ushort2)(bv12, bv13));
            b0.s7 = as_int((ushort2)(bv14, bv15));
        }

        short8 a00, a01, a02, a03;
        {
            __local const half* a_k0 = a_slm + sg_lid;
            a00.s0 = as_short(a_k0[0 * A_SLM_STRIDE]);
            a00.s1 = as_short(a_k0[1 * A_SLM_STRIDE]);
            a00.s2 = as_short(a_k0[2 * A_SLM_STRIDE]);
            a00.s3 = as_short(a_k0[3 * A_SLM_STRIDE]);
            a00.s4 = as_short(a_k0[4 * A_SLM_STRIDE]);
            a00.s5 = as_short(a_k0[5 * A_SLM_STRIDE]);
            a00.s6 = as_short(a_k0[6 * A_SLM_STRIDE]);
            a00.s7 = as_short(a_k0[7 * A_SLM_STRIDE]);

            a01.s0 = as_short(a_k0[8 * A_SLM_STRIDE]);
            a01.s1 = as_short(a_k0[9 * A_SLM_STRIDE]);
            a01.s2 = as_short(a_k0[10 * A_SLM_STRIDE]);
            a01.s3 = as_short(a_k0[11 * A_SLM_STRIDE]);
            a01.s4 = as_short(a_k0[12 * A_SLM_STRIDE]);
            a01.s5 = as_short(a_k0[13 * A_SLM_STRIDE]);
            a01.s6 = as_short(a_k0[14 * A_SLM_STRIDE]);
            a01.s7 = as_short(a_k0[15 * A_SLM_STRIDE]);

            a02.s0 = as_short(a_k0[16 * A_SLM_STRIDE]);
            a02.s1 = as_short(a_k0[17 * A_SLM_STRIDE]);
            a02.s2 = as_short(a_k0[18 * A_SLM_STRIDE]);
            a02.s3 = as_short(a_k0[19 * A_SLM_STRIDE]);
            a02.s4 = as_short(a_k0[20 * A_SLM_STRIDE]);
            a02.s5 = as_short(a_k0[21 * A_SLM_STRIDE]);
            a02.s6 = as_short(a_k0[22 * A_SLM_STRIDE]);
            a02.s7 = as_short(a_k0[23 * A_SLM_STRIDE]);

            a03.s0 = as_short(a_k0[24 * A_SLM_STRIDE]);
            a03.s1 = as_short(a_k0[25 * A_SLM_STRIDE]);
            a03.s2 = as_short(a_k0[26 * A_SLM_STRIDE]);
            a03.s3 = as_short(a_k0[27 * A_SLM_STRIDE]);
            a03.s4 = as_short(a_k0[28 * A_SLM_STRIDE]);
            a03.s5 = as_short(a_k0[29 * A_SLM_STRIDE]);
            a03.s6 = as_short(a_k0[30 * A_SLM_STRIDE]);
            a03.s7 = as_short(a_k0[31 * A_SLM_STRIDE]);
        }

        acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a00, b0, acc0);
        acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a01, b0, acc1);
        acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a02, b0, acc2);
        acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a03, b0, acc3);

        // Second k16
        int8 b1;
        {
            __global const half* b_ptr = B + (k_base + 16) * N + baseN + sg_lid;
            ushort bv0  = as_ushort(b_ptr[0 * N]);
            ushort bv1  = as_ushort(b_ptr[1 * N]);
            ushort bv2  = as_ushort(b_ptr[2 * N]);
            ushort bv3  = as_ushort(b_ptr[3 * N]);
            ushort bv4  = as_ushort(b_ptr[4 * N]);
            ushort bv5  = as_ushort(b_ptr[5 * N]);
            ushort bv6  = as_ushort(b_ptr[6 * N]);
            ushort bv7  = as_ushort(b_ptr[7 * N]);
            ushort bv8  = as_ushort(b_ptr[8 * N]);
            ushort bv9  = as_ushort(b_ptr[9 * N]);
            ushort bv10 = as_ushort(b_ptr[10 * N]);
            ushort bv11 = as_ushort(b_ptr[11 * N]);
            ushort bv12 = as_ushort(b_ptr[12 * N]);
            ushort bv13 = as_ushort(b_ptr[13 * N]);
            ushort bv14 = as_ushort(b_ptr[14 * N]);
            ushort bv15 = as_ushort(b_ptr[15 * N]);
            b1.s0 = as_int((ushort2)(bv0, bv1));
            b1.s1 = as_int((ushort2)(bv2, bv3));
            b1.s2 = as_int((ushort2)(bv4, bv5));
            b1.s3 = as_int((ushort2)(bv6, bv7));
            b1.s4 = as_int((ushort2)(bv8, bv9));
            b1.s5 = as_int((ushort2)(bv10, bv11));
            b1.s6 = as_int((ushort2)(bv12, bv13));
            b1.s7 = as_int((ushort2)(bv14, bv15));
        }

        short8 a10, a11, a12, a13;
        {
            __local const half* a_k1 = a_slm + 16 + sg_lid;
            a10.s0 = as_short(a_k1[0 * A_SLM_STRIDE]);
            a10.s1 = as_short(a_k1[1 * A_SLM_STRIDE]);
            a10.s2 = as_short(a_k1[2 * A_SLM_STRIDE]);
            a10.s3 = as_short(a_k1[3 * A_SLM_STRIDE]);
            a10.s4 = as_short(a_k1[4 * A_SLM_STRIDE]);
            a10.s5 = as_short(a_k1[5 * A_SLM_STRIDE]);
            a10.s6 = as_short(a_k1[6 * A_SLM_STRIDE]);
            a10.s7 = as_short(a_k1[7 * A_SLM_STRIDE]);

            a11.s0 = as_short(a_k1[8 * A_SLM_STRIDE]);
            a11.s1 = as_short(a_k1[9 * A_SLM_STRIDE]);
            a11.s2 = as_short(a_k1[10 * A_SLM_STRIDE]);
            a11.s3 = as_short(a_k1[11 * A_SLM_STRIDE]);
            a11.s4 = as_short(a_k1[12 * A_SLM_STRIDE]);
            a11.s5 = as_short(a_k1[13 * A_SLM_STRIDE]);
            a11.s6 = as_short(a_k1[14 * A_SLM_STRIDE]);
            a11.s7 = as_short(a_k1[15 * A_SLM_STRIDE]);

            a12.s0 = as_short(a_k1[16 * A_SLM_STRIDE]);
            a12.s1 = as_short(a_k1[17 * A_SLM_STRIDE]);
            a12.s2 = as_short(a_k1[18 * A_SLM_STRIDE]);
            a12.s3 = as_short(a_k1[19 * A_SLM_STRIDE]);
            a12.s4 = as_short(a_k1[20 * A_SLM_STRIDE]);
            a12.s5 = as_short(a_k1[21 * A_SLM_STRIDE]);
            a12.s6 = as_short(a_k1[22 * A_SLM_STRIDE]);
            a12.s7 = as_short(a_k1[23 * A_SLM_STRIDE]);

            a13.s0 = as_short(a_k1[24 * A_SLM_STRIDE]);
            a13.s1 = as_short(a_k1[25 * A_SLM_STRIDE]);
            a13.s2 = as_short(a_k1[26 * A_SLM_STRIDE]);
            a13.s3 = as_short(a_k1[27 * A_SLM_STRIDE]);
            a13.s4 = as_short(a_k1[28 * A_SLM_STRIDE]);
            a13.s5 = as_short(a_k1[29 * A_SLM_STRIDE]);
            a13.s6 = as_short(a_k1[30 * A_SLM_STRIDE]);
            a13.s7 = as_short(a_k1[31 * A_SLM_STRIDE]);
        }

        acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a10, b1, acc0);
        acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a11, b1, acc1);
        acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a12, b1, acc2);
        acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a13, b1, acc3);
    }

    // Store C: each subgroup writes 32 rows x 16 cols
    __global half* c_out = C + baseM * N + baseN + sg_lid;

    #pragma unroll
    for (int r = 0; r < 8; r++) {
        c_out[(0 + r) * N] = convert_half(((float*)&acc0)[r]);
    }
    #pragma unroll
    for (int r = 0; r < 8; r++) {
        c_out[(8 + r) * N] = convert_half(((float*)&acc1)[r]);
    }
    #pragma unroll
    for (int r = 0; r < 8; r++) {
        c_out[(16 + r) * N] = convert_half(((float*)&acc2)[r]);
    }
    #pragma unroll
    for (int r = 0; r < 8; r++) {
        c_out[(24 + r) * N] = convert_half(((float*)&acc3)[r]);
    }

    #undef A_SLM_STRIDE
    #undef A_SLM_SIZE
}
```

### Previous kernel and evaluation:
Here is the last OCL kernel we tested (Result: Correctness score: 5 / 5 (compiles and correct), runtime in ms: 0.721):
```OCL
// GEMM: C[M,N] = A[M,K] x B[K,N], row-major, FP16 in/out, FP32 accumulation
// Architecture: 64 WIs (4 SGs of 16), TILE 32x64x32, A in SLM (double-buffered), B from global
// Launch: GWS = (N/64 * 64, M/32), LWS = (64, 1)
// Each WG computes a 32(M) x 64(N) output tile
// 4 subgroups, each handles 32 rows x 16 cols
// K-loop unrolled 2x: processes 64 K-elements per iteration (32 iterations for K=2048)

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void matmul(
    __global const half* restrict A,
    __global const half* restrict B,
    __global half* restrict C,
    const int M,
    const int K,
    const int N)
{
    const int wg_n = get_group_id(0);
    const int wg_m = get_group_id(1);

    const int lid = get_local_id(0);
    const int sg_id = lid / 16;
    const int sg_lid = get_sub_group_local_id();

    const int baseM = wg_m * 32;
    const int baseN = wg_n * 64 + sg_id * 16;

    // SLM for A: double-buffered, 32 rows x 32 cols, stride=32
    #define A_SLM_STRIDE 32
    #define A_SLM_SIZE (32 * A_SLM_STRIDE)
    __local half slm_A[2 * A_SLM_SIZE];

    // Accumulators
    float8 acc0 = (float8)(0.0f);
    float8 acc1 = (float8)(0.0f);
    float8 acc2 = (float8)(0.0f);
    float8 acc3 = (float8)(0.0f);

    // Cooperative A load: 64 WIs load 32x32 = 1024 halfs = 16 halfs/WI
    const int a_load_row = lid / 2;
    const int a_load_col = (lid & 1) * 16;
    const int a_gm_row = baseM + a_load_row;

    // Preload first A tile (k=0..31) into SLM buffer 0
    {
        __global const half* a_src = A + a_gm_row * K + a_load_col;
        __local half* a_dst = slm_A + a_load_row * A_SLM_STRIDE + a_load_col;
        *(__local half16*)a_dst = *(__global const half16*)a_src;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // K=2048, step=32 -> 64 iterations. We do NOT unroll 2x to keep register pressure low.
    // Instead we focus on better interleaving of loads and compute.
    const int k_iters = K / 32;  // 64
    int cur_buf = 0;

    // Main loop: all iterations except the last
    for (int ki = 0; ki < k_iters - 1; ki++) {
        const int k_base = ki * 32;
        const int next_buf = 1 - cur_buf;

        // Pointer to current A buffer in SLM
        __local const half* a_slm = slm_A + cur_buf * A_SLM_SIZE;

        // === First k16 step: k_base..k_base+15 ===

        // Load B for first k16
        int8 b0;
        {
            __global const half* b_col = B + k_base * N + baseN + sg_lid;
            ushort bv0  = as_ushort(b_col[0 * N]);
            ushort bv1  = as_ushort(b_col[1 * N]);
            ushort bv2  = as_ushort(b_col[2 * N]);
            ushort bv3  = as_ushort(b_col[3 * N]);
            ushort bv4  = as_ushort(b_col[4 * N]);
            ushort bv5  = as_ushort(b_col[5 * N]);
            ushort bv6  = as_ushort(b_col[6 * N]);
            ushort bv7  = as_ushort(b_col[7 * N]);
            ushort bv8  = as_ushort(b_col[8 * N]);
            ushort bv9  = as_ushort(b_col[9 * N]);
            ushort bv10 = as_ushort(b_col[10 * N]);
            ushort bv11 = as_ushort(b_col[11 * N]);
            ushort bv12 = as_ushort(b_col[12 * N]);
            ushort bv13 = as_ushort(b_col[13 * N]);
            ushort bv14 = as_ushort(b_col[14 * N]);
            ushort bv15 = as_ushort(b_col[15 * N]);

            b0.s0 = as_int((ushort2)(bv0, bv1));
            b0.s1 = as_int((ushort2)(bv2, bv3));
            b0.s2 = as_int((ushort2)(bv4, bv5));
            b0.s3 = as_int((ushort2)(bv6, bv7));
            b0.s4 = as_int((ushort2)(bv8, bv9));
            b0.s5 = as_int((ushort2)(bv10, bv11));
            b0.s6 = as_int((ushort2)(bv12, bv13));
            b0.s7 = as_int((ushort2)(bv14, bv15));
        }

        // Load A from SLM for first k16
        short8 a00, a01, a02, a03;
        {
            __local const half* a_k0 = a_slm + sg_lid;
            a00.s0 = as_short(a_k0[0 * A_SLM_STRIDE]);
            a00.s1 = as_short(a_k0[1 * A_SLM_STRIDE]);
            a00.s2 = as_short(a_k0[2 * A_SLM_STRIDE]);
            a00.s3 = as_short(a_k0[3 * A_SLM_STRIDE]);
            a00.s4 = as_short(a_k0[4 * A_SLM_STRIDE]);
            a00.s5 = as_short(a_k0[5 * A_SLM_STRIDE]);
            a00.s6 = as_short(a_k0[6 * A_SLM_STRIDE]);
            a00.s7 = as_short(a_k0[7 * A_SLM_STRIDE]);

            a01.s0 = as_short(a_k0[8 * A_SLM_STRIDE]);
            a01.s1 = as_short(a_k0[9 * A_SLM_STRIDE]);
            a01.s2 = as_short(a_k0[10 * A_SLM_STRIDE]);
            a01.s3 = as_short(a_k0[11 * A_SLM_STRIDE]);
            a01.s4 = as_short(a_k0[12 * A_SLM_STRIDE]);
            a01.s5 = as_short(a_k0[13 * A_SLM_STRIDE]);
            a01.s6 = as_short(a_k0[14 * A_SLM_STRIDE]);
            a01.s7 = as_short(a_k0[15 * A_SLM_STRIDE]);

            a02.s0 = as_short(a_k0[16 * A_SLM_STRIDE]);
            a02.s1 = as_short(a_k0[17 * A_SLM_STRIDE]);
            a02.s2 = as_short(a_k0[18 * A_SLM_STRIDE]);
            a02.s3 = as_short(a_k0[19 * A_SLM_STRIDE]);
            a02.s4 = as_short(a_k0[20 * A_SLM_STRIDE]);
            a02.s5 = as_short(a_k0[21 * A_SLM_STRIDE]);
            a02.s6 = as_short(a_k0[22 * A_SLM_STRIDE]);
            a02.s7 = as_short(a_k0[23 * A_SLM_STRIDE]);

            a03.s0 = as_short(a_k0[24 * A_SLM_STRIDE]);
            a03.s1 = as_short(a_k0[25 * A_SLM_STRIDE]);
            a03.s2 = as_short(a_k0[26 * A_SLM_STRIDE]);
            a03.s3 = as_short(a_k0[27 * A_SLM_STRIDE]);
            a03.s4 = as_short(a_k0[28 * A_SLM_STRIDE]);
            a03.s5 = as_short(a_k0[29 * A_SLM_STRIDE]);
            a03.s6 = as_short(a_k0[30 * A_SLM_STRIDE]);
            a03.s7 = as_short(a_k0[31 * A_SLM_STRIDE]);
        }

        // DPAS first k16
        acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a00, b0, acc0);
        acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a01, b0, acc1);

        // Interleave: start loading next A tile while DPAS is in flight
        // This overlaps the global memory load with DPAS execution
        __global const half* a_next_src = A + a_gm_row * K + (k_base + 32) + a_load_col;
        __local half* a_next_dst = slm_A + next_buf * A_SLM_SIZE + a_load_row * A_SLM_STRIDE + a_load_col;
        *(__local half16*)a_next_dst = *(__global const half16*)a_next_src;

        acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a02, b0, acc2);
        acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a03, b0, acc3);

        // === Second k16 step: k_base+16..k_base+31 ===

        // Load B for second k16
        int8 b1;
        {
            __global const half* b_col = B + (k_base + 16) * N + baseN + sg_lid;
            ushort bv0  = as_ushort(b_col[0 * N]);
            ushort bv1  = as_ushort(b_col[1 * N]);
            ushort bv2  = as_ushort(b_col[2 * N]);
            ushort bv3  = as_ushort(b_col[3 * N]);
            ushort bv4  = as_ushort(b_col[4 * N]);
            ushort bv5  = as_ushort(b_col[5 * N]);
            ushort bv6  = as_ushort(b_col[6 * N]);
            ushort bv7  = as_ushort(b_col[7 * N]);
            ushort bv8  = as_ushort(b_col[8 * N]);
            ushort bv9  = as_ushort(b_col[9 * N]);
            ushort bv10 = as_ushort(b_col[10 * N]);
            ushort bv11 = as_ushort(b_col[11 * N]);
            ushort bv12 = as_ushort(b_col[12 * N]);
            ushort bv13 = as_ushort(b_col[13 * N]);
            ushort bv14 = as_ushort(b_col[14 * N]);
            ushort bv15 = as_ushort(b_col[15 * N]);

            b1.s0 = as_int((ushort2)(bv0, bv1));
            b1.s1 = as_int((ushort2)(bv2, bv3));
            b1.s2 = as_int((ushort2)(bv4, bv5));
            b1.s3 = as_int((ushort2)(bv6, bv7));
            b1.s4 = as_int((ushort2)(bv8, bv9));
            b1.s5 = as_int((ushort2)(bv10, bv11));
            b1.s6 = as_int((ushort2)(bv12, bv13));
            b1.s7 = as_int((ushort2)(bv14, bv15));
        }

        // Load A from SLM for second k16
        short8 a10, a11, a12, a13;
        {
            __local const half* a_k1 = a_slm + 16 + sg_lid;
            a10.s0 = as_short(a_k1[0 * A_SLM_STRIDE]);
            a10.s1 = as_short(a_k1[1 * A_SLM_STRIDE]);
            a10.s2 = as_short(a_k1[2 * A_SLM_STRIDE]);
            a10.s3 = as_short(a_k1[3 * A_SLM_STRIDE]);
            a10.s4 = as_short(a_k1[4 * A_SLM_STRIDE]);
            a10.s5 = as_short(a_k1[5 * A_SLM_STRIDE]);
            a10.s6 = as_short(a_k1[6 * A_SLM_STRIDE]);
            a10.s7 = as_short(a_k1[7 * A_SLM_STRIDE]);

            a11.s0 = as_short(a_k1[8 * A_SLM_STRIDE]);
            a11.s1 = as_short(a_k1[9 * A_SLM_STRIDE]);
            a11.s2 = as_short(a_k1[10 * A_SLM_STRIDE]);
            a11.s3 = as_short(a_k1[11 * A_SLM_STRIDE]);
            a11.s4 = as_short(a_k1[12 * A_SLM_STRIDE]);
            a11.s5 = as_short(a_k1[13 * A_SLM_STRIDE]);
            a11.s6 = as_short(a_k1[14 * A_SLM_STRIDE]);
            a11.s7 = as_short(a_k1[15 * A_SLM_STRIDE]);

            a12.s0 = as_short(a_k1[16 * A_SLM_STRIDE]);
            a12.s1 = as_short(a_k1[17 * A_SLM_STRIDE]);
            a12.s2 = as_short(a_k1[18 * A_SLM_STRIDE]);
            a12.s3 = as_short(a_k1[19 * A_SLM_STRIDE]);
            a12.s4 = as_short(a_k1[20 * A_SLM_STRIDE]);
            a12.s5 = as_short(a_k1[21 * A_SLM_STRIDE]);
            a12.s6 = as_short(a_k1[22 * A_SLM_STRIDE]);
            a12.s7 = as_short(a_k1[23 * A_SLM_STRIDE]);

            a13.s0 = as_short(a_k1[24 * A_SLM_STRIDE]);
            a13.s1 = as_short(a_k1[25 * A_SLM_STRIDE]);
            a13.s2 = as_short(a_k1[26 * A_SLM_STRIDE]);
            a13.s3 = as_short(a_k1[27 * A_SLM_STRIDE]);
            a13.s4 = as_short(a_k1[28 * A_SLM_STRIDE]);
            a13.s5 = as_short(a_k1[29 * A_SLM_STRIDE]);
            a13.s6 = as_short(a_k1[30 * A_SLM_STRIDE]);
            a13.s7 = as_short(a_k1[31 * A_SLM_STRIDE]);
        }

        // DPAS second k16
        acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a10, b1, acc0);
        acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a11, b1, acc1);
        acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a12, b1, acc2);
        acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a13, b1, acc3);

        cur_buf = next_buf;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // === Last iteration (ki = k_iters - 1): no next A load needed ===
    {
        const int k_base = (k_iters - 1) * 32;
        __local const half* a_slm = slm_A + cur_buf * A_SLM_SIZE;

        // First k16
        int8 b0;
        {
            __global const half* b_col = B + k_base * N + baseN + sg_lid;
            ushort bv0  = as_ushort(b_col[0 * N]);
            ushort bv1  = as_ushort(b_col[1 * N]);
            ushort bv2  = as_ushort(b_col[2 * N]);
            ushort bv3  = as_ushort(b_col[3 * N]);
            ushort bv4  = as_ushort(b_col[4 * N]);
            ushort bv5  = as_ushort(b_col[5 * N]);
            ushort bv6  = as_ushort(b_col[6 * N]);
            ushort bv7  = as_ushort(b_col[7 * N]);
            ushort bv8  = as_ushort(b_col[8 * N]);
            ushort bv9  = as_ushort(b_col[9 * N]);
            ushort bv10 = as_ushort(b_col[10 * N]);
            ushort bv11 = as_ushort(b_col[11 * N]);
            ushort bv12 = as_ushort(b_col[12 * N]);
            ushort bv13 = as_ushort(b_col[13 * N]);
            ushort bv14 = as_ushort(b_col[14 * N]);
            ushort bv15 = as_ushort(b_col[15 * N]);

            b0.s0 = as_int((ushort2)(bv0, bv1));
            b0.s1 = as_int((ushort2)(bv2, bv3));
            b0.s2 = as_int((ushort2)(bv4, bv5));
            b0.s3 = as_int((ushort2)(bv6, bv7));
            b0.s4 = as_int((ushort2)(bv8, bv9));
            b0.s5 = as_int((ushort2)(bv10, bv11));
            b0.s6 = as_int((ushort2)(bv12, bv13));
            b0.s7 = as_int((ushort2)(bv14, bv15));
        }

        short8 a00, a01, a02, a03;
        {
            __local const half* a_k0 = a_slm + sg_lid;
            a00.s0 = as_short(a_k0[0 * A_SLM_STRIDE]);
            a00.s1 = as_short(a_k0[1 * A_SLM_STRIDE]);
            a00.s2 = as_short(a_k0[2 * A_SLM_STRIDE]);
            a00.s3 = as_short(a_k0[3 * A_SLM_STRIDE]);
            a00.s4 = as_short(a_k0[4 * A_SLM_STRIDE]);
            a00.s5 = as_short(a_k0[5 * A_SLM_STRIDE]);
            a00.s6 = as_short(a_k0[6 * A_SLM_STRIDE]);
            a00.s7 = as_short(a_k0[7 * A_SLM_STRIDE]);

            a01.s0 = as_short(a_k0[8 * A_SLM_STRIDE]);
            a01.s1 = as_short(a_k0[9 * A_SLM_STRIDE]);
            a01.s2 = as_short(a_k0[10 * A_SLM_STRIDE]);
            a01.s3 = as_short(a_k0[11 * A_SLM_STRIDE]);
            a01.s4 = as_short(a_k0[12 * A_SLM_STRIDE]);
            a01.s5 = as_short(a_k0[13 * A_SLM_STRIDE]);
            a01.s6 = as_short(a_k0[14 * A_SLM_STRIDE]);
            a01.s7 = as_short(a_k0[15 * A_SLM_STRIDE]);

            a02.s0 = as_short(a_k0[16 * A_SLM_STRIDE]);
            a02.s1 = as_short(a_k0[17 * A_SLM_STRIDE]);
            a02.s2 = as_short(a_k0[18 * A_SLM_STRIDE]);
            a02.s3 = as_short(a_k0[19 * A_SLM_STRIDE]);
            a02.s4 = as_short(a_k0[20 * A_SLM_STRIDE]);
            a02.s5 = as_short(a_k0[21 * A_SLM_STRIDE]);
            a02.s6 = as_short(a_k0[22 * A_SLM_STRIDE]);
            a02.s7 = as_short(a_k0[23 * A_SLM_STRIDE]);

            a03.s0 = as_short(a_k0[24 * A_SLM_STRIDE]);
            a03.s1 = as_short(a_k0[25 * A_SLM_STRIDE]);
            a03.s2 = as_short(a_k0[26 * A_SLM_STRIDE]);
            a03.s3 = as_short(a_k0[27 * A_SLM_STRIDE]);
            a03.s4 = as_short(a_k0[28 * A_SLM_STRIDE]);
            a03.s5 = as_short(a_k0[29 * A_SLM_STRIDE]);
            a03.s6 = as_short(a_k0[30 * A_SLM_STRIDE]);
            a03.s7 = as_short(a_k0[31 * A_SLM_STRIDE]);
        }

        acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a00, b0, acc0);
        acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a01, b0, acc1);
        acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a02, b0, acc2);
        acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a03, b0, acc3);

        // Second k16
        int8 b1;
        {
            __global const half* b_col = B + (k_base + 16) * N + baseN + sg_lid;
            ushort bv0  = as_ushort(b_col[0 * N]);
            ushort bv1  = as_ushort(b_col[1 * N]);
            ushort bv2  = as_ushort(b_col[2 * N]);
            ushort bv3  = as_ushort(b_col[3 * N]);
            ushort bv4  = as_ushort(b_col[4 * N]);
            ushort bv5  = as_ushort(b_col[5 * N]);
            ushort bv6  = as_ushort(b_col[6 * N]);
            ushort bv7  = as_ushort(b_col[7 * N]);
            ushort bv8  = as_ushort(b_col[8 * N]);
            ushort bv9  = as_ushort(b_col[9 * N]);
            ushort bv10 = as_ushort(b_col[10 * N]);
            ushort bv11 = as_ushort(b_col[11 * N]);
            ushort bv12 = as_ushort(b_col[12 * N]);
            ushort bv13 = as_ushort(b_col[13 * N]);
            ushort bv14 = as_ushort(b_col[14 * N]);
            ushort bv15 = as_ushort(b_col[15 * N]);

            b1.s0 = as_int((ushort2)(bv0, bv1));
            b1.s1 = as_int((ushort2)(bv2, bv3));
            b1.s2 = as_int((ushort2)(bv4, bv5));
            b1.s3 = as_int((ushort2)(bv6, bv7));
            b1.s4 = as_int((ushort2)(bv8, bv9));
            b1.s5 = as_int((ushort2)(bv10, bv11));
            b1.s6 = as_int((ushort2)(bv12, bv13));
            b1.s7 = as_int((ushort2)(bv14, bv15));
        }

        short8 a10, a11, a12, a13;
        {
            __local const half* a_k1 = a_slm + 16 + sg_lid;
            a10.s0 = as_short(a_k1[0 * A_SLM_STRIDE]);
            a10.s1 = as_short(a_k1[1 * A_SLM_STRIDE]);
            a10.s2 = as_short(a_k1[2 * A_SLM_STRIDE]);
            a10.s3 = as_short(a_k1[3 * A_SLM_STRIDE]);
            a10.s4 = as_short(a_k1[4 * A_SLM_STRIDE]);
            a10.s5 = as_short(a_k1[5 * A_SLM_STRIDE]);
            a10.s6 = as_short(a_k1[6 * A_SLM_STRIDE]);
            a10.s7 = as_short(a_k1[7 * A_SLM_STRIDE]);

            a11.s0 = as_short(a_k1[8 * A_SLM_STRIDE]);
            a11.s1 = as_short(a_k1[9 * A_SLM_STRIDE]);
            a11.s2 = as_short(a_k1[10 * A_SLM_STRIDE]);
            a11.s3 = as_short(a_k1[11 * A_SLM_STRIDE]);
            a11.s4 = as_short(a_k1[12 * A_SLM_STRIDE]);
            a11.s5 = as_short(a_k1[13 * A_SLM_STRIDE]);
            a11.s6 = as_short(a_k1[14 * A_SLM_STRIDE]);
            a11.s7 = as_short(a_k1[15 * A_SLM_STRIDE]);

            a12.s0 = as_short(a_k1[16 * A_SLM_STRIDE]);
            a12.s1 = as_short(a_k1[17 * A_SLM_STRIDE]);
            a12.s2 = as_short(a_k1[18 * A_SLM_STRIDE]);
            a12.s3 = as_short(a_k1[19 * A_SLM_STRIDE]);
            a12.s4 = as_short(a_k1[20 * A_SLM_STRIDE]);
            a12.s5 = as_short(a_k1[21 * A_SLM_STRIDE]);
            a12.s6 = as_short(a_k1[22 * A_SLM_STRIDE]);
            a12.s7 = as_short(a_k1[23 * A_SLM_STRIDE]);

            a13.s0 = as_short(a_k1[24 * A_SLM_STRIDE]);
            a13.s1 = as_short(a_k1[25 * A_SLM_STRIDE]);
            a13.s2 = as_short(a_k1[26 * A_SLM_STRIDE]);
            a13.s3 = as_short(a_k1[27 * A_SLM_STRIDE]);
            a13.s4 = as_short(a_k1[28 * A_SLM_STRIDE]);
            a13.s5 = as_short(a_k1[29 * A_SLM_STRIDE]);
            a13.s6 = as_short(a_k1[30 * A_SLM_STRIDE]);
            a13.s7 = as_short(a_k1[31 * A_SLM_STRIDE]);
        }

        acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a10, b1, acc0);
        acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a11, b1, acc1);
        acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a12, b1, acc2);
        acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a13, b1, acc3);
    }

    // Store C: each subgroup writes 32 rows x 16 cols (one half per WI per row)
    __global half* c_out = C + baseM * N + baseN + sg_lid;

    #pragma unroll
    for (int r = 0; r < 8; r++) {
        c_out[(0 + r) * N] = convert_half(((float*)&acc0)[r]);
    }
    #pragma unroll
    for (int r = 0; r < 8; r++) {
        c_out[(8 + r) * N] = convert_half(((float*)&acc1)[r]);
    }
    #pragma unroll
    for (int r = 0; r < 8; r++) {
        c_out[(16 + r) * N] = convert_half(((float*)&acc2)[r]);
    }
    #pragma unroll
    for (int r = 0; r < 8; r++) {
        c_out[(24 + r) * N] = convert_half(((float*)&acc3)[r]);
    }

    #undef A_SLM_STRIDE
    #undef A_SLM_SIZE
}
```

Console output from running this kernel:

Test result on platform Intel Corporation Battlemage G21 [Intel Graphics]:
==== test session starts

task.py::TestMatmulOCL::test_correctness_wrt_pytorch[0] PASSED           [ 25%]
task.py::TestMatmulOCL::test_correctness_wrt_pytorch[1] PASSED           [ 50%]
task.py::TestMatmulOCL::test_correctness_wrt_reference[0] PASSED         [ 75%]
task.py::TestMatmulOCL::test_correctness_wrt_reference[1] PASSED         [100%]

======================= 4 passed, 1 deselected in 0.99s ========================
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
1. Maximize Occupancy: Balance work-group size with register/local memory usage. Query CL_KERNEL_WORK_GROUP_SIZE and CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE. Typical work-group sizes: 64-256 for compute-bound, 32-128 for memory-bound kernels. Use multiples of wavefront/warp size (32 for NVIDIA, 64 for AMD).
2. Async Memory Operations: Use async_work_group_copy() to overlap computation with memory transfers between global and local memory. Use async_work_group_strided_copy() for non-contiguous data. Wait with wait_group_events().

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
- **Hierarchical Parallelism**: Structure work at three levels - work-groups (tile assignment), sub-groups (sub-tile processing), work-items (register tile). Use sub-group shuffles to share data within sub-group without SLM.