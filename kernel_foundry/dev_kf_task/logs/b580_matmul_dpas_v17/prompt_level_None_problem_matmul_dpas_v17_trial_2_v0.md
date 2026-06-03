

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

### Version 1 (Result: Correctness score: 5 / 5 (compiles and correct), runtime in ms: 1.300):
```OCL
// GEMM: C[M,N] = A[M,K] x B[K,N], row-major, FP16 in/out, FP32 accumulation
// Architecture: 64 WIs (4 SGs of 16), TILE 32x64x32, A in SLM (double-buffered), B from global
// Launch: GWS = ((N/64)*16, (M/32)*4), LWS = (16, 4)
// Each WG computes a 32-row x 64-col output tile
// Each subgroup computes 32 rows x 16 cols
// Subgroup size = 16
__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(16, 4, 1)))
__kernel void matmul(
    __global const half* A,
    __global const half* B,
    __global half* C,
    const int M,
    const int K,
    const int N)
{
    // Work-group tile indices
    const int wg_n = get_group_id(0);  // which 64-col tile (0 .. N/64-1)
    const int wg_m = get_group_id(1);  // which 32-row tile (0 .. M/32-1)

    const int sg_id = get_local_id(1);           // subgroup index 0..3
    const int sg_lid = get_sub_group_local_id(); // lane within subgroup 0..15

    // Base row/col for this WG's output tile
    const int baseM = wg_m * 32;
    const int baseN = wg_n * 64 + sg_id * 16;  // each SG handles 16 columns

    // SLM for A: double-buffered, 32 rows x 32 cols
    // Stride = 32 (no padding needed since we access with stride, not bank-conflicting pattern)
    // Actually use stride=34 for bank conflict avoidance
    #define SLM_A_STRIDE 34
    __local half slm_A[2 * 32 * SLM_A_STRIDE];

    // Accumulators: 32 rows x 16 cols = 4 groups of 8 rows
    float8 acc0 = (float8)(0.0f);  // rows 0-7
    float8 acc1 = (float8)(0.0f);  // rows 8-15
    float8 acc2 = (float8)(0.0f);  // rows 16-23
    float8 acc3 = (float8)(0.0f);  // rows 24-31

    // Linear local ID for cooperative A loading
    const int lid = sg_id * 16 + sg_lid;  // 0..63

    // Cooperative A load mapping: 64 WIs load 32x32 = 1024 halfs = 16 halfs per WI
    // Each WI loads 16 consecutive halfs in K-dimension for one row
    // lid / 2 = row (0..31), (lid % 2) * 16 = col offset (0 or 16)
    const int a_load_row = lid / 2;
    const int a_load_col = (lid % 2) * 16;

    // Preload first A tile (k=0..31) into SLM buffer 0
    {
        int global_row = baseM + a_load_row;
        __global const half* a_src = A + global_row * K + a_load_col;
        __local half* slm_dst = slm_A + a_load_row * SLM_A_STRIDE + a_load_col;

        #pragma unroll
        for (int i = 0; i < 16; i++) {
            slm_dst[i] = a_src[i];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int buf = 0;
    const int k_iters = K / 32;  // K=2048 -> 64 iterations

    for (int ki = 0; ki < k_iters; ki++) {
        int k_base = ki * 32;

        // Pointer to current SLM buffer for A
        __local const half* cur_slm = slm_A + buf * 32 * SLM_A_STRIDE;

        // === Load B from global memory ===
        // B is [K, N] row-major. We need B[k_base..k_base+31, baseN..baseN+15]
        // For DPAS int8 b operand with k16:
        //   int8 per WI, WI sg_lid reads column sg_lid
        //   Each int packs 2 consecutive k-values: b.s0 = pack(B[k+0,col], B[k+1,col])
        //   int8 = 8 ints = 16 k-values

        // B for first k16 step (k_base+0..k_base+15)
        int8 b_val0;
        {
            __global const half* b_col = B + k_base * N + baseN + sg_lid;
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                ushort lo = as_ushort(b_col[(i * 2) * N]);
                ushort hi = as_ushort(b_col[(i * 2 + 1) * N]);
                ((int*)&b_val0)[i] = (int)((uint)lo | ((uint)hi << 16));
            }
        }

        // B for second k16 step (k_base+16..k_base+31)
        int8 b_val1;
        {
            __global const half* b_col = B + (k_base + 16) * N + baseN + sg_lid;
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                ushort lo = as_ushort(b_col[(i * 2) * N]);
                ushort hi = as_ushort(b_col[(i * 2 + 1) * N]);
                ((int*)&b_val1)[i] = (int)((uint)lo | ((uint)hi << 16));
            }
        }

        // === Load A from SLM ===
        // For DPAS short8 a operand with k16:
        //   short8 per WI, representing 8 rows x 16 K-elements
        //   WI sg_lid holds: a.s_r = A[row_base + r, k_base + sg_lid] as short (= half bits)
        //   So each element of short8 is one row's k-element at position sg_lid

        // First k16 step: read from SLM columns 0..15
        short8 a00, a01, a02, a03;
        #pragma unroll
        for (int r = 0; r < 8; r++) {
            ((short*)&a00)[r] = as_short(cur_slm[(0 + r) * SLM_A_STRIDE + sg_lid]);
        }
        #pragma unroll
        for (int r = 0; r < 8; r++) {
            ((short*)&a01)[r] = as_short(cur_slm[(8 + r) * SLM_A_STRIDE + sg_lid]);
        }
        #pragma unroll
        for (int r = 0; r < 8; r++) {
            ((short*)&a02)[r] = as_short(cur_slm[(16 + r) * SLM_A_STRIDE + sg_lid]);
        }
        #pragma unroll
        for (int r = 0; r < 8; r++) {
            ((short*)&a03)[r] = as_short(cur_slm[(24 + r) * SLM_A_STRIDE + sg_lid]);
        }

        // DPAS for first k16
        acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a00, b_val0, acc0);
        acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a01, b_val0, acc1);
        acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a02, b_val0, acc2);
        acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a03, b_val0, acc3);

        // Second k16 step: read from SLM columns 16..31
        short8 a10, a11, a12, a13;
        #pragma unroll
        for (int r = 0; r < 8; r++) {
            ((short*)&a10)[r] = as_short(cur_slm[(0 + r) * SLM_A_STRIDE + 16 + sg_lid]);
        }
        #pragma unroll
        for (int r = 0; r < 8; r++) {
            ((short*)&a11)[r] = as_short(cur_slm[(8 + r) * SLM_A_STRIDE + 16 + sg_lid]);
        }
        #pragma unroll
        for (int r = 0; r < 8; r++) {
            ((short*)&a12)[r] = as_short(cur_slm[(16 + r) * SLM_A_STRIDE + 16 + sg_lid]);
        }
        #pragma unroll
        for (int r = 0; r < 8; r++) {
            ((short*)&a13)[r] = as_short(cur_slm[(24 + r) * SLM_A_STRIDE + 16 + sg_lid]);
        }

        // DPAS for second k16
        acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a10, b_val1, acc0);
        acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a11, b_val1, acc1);
        acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a12, b_val1, acc2);
        acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a13, b_val1, acc3);

        // === Load next A tile into other SLM buffer (if not last iteration) ===
        if (ki < k_iters - 1) {
            int next_buf = 1 - buf;
            int next_k = (ki + 1) * 32;
            int global_row = baseM + a_load_row;
            __global const half* a_src = A + global_row * K + next_k + a_load_col;
            __local half* slm_dst = slm_A + next_buf * 32 * SLM_A_STRIDE + a_load_row * SLM_A_STRIDE + a_load_col;

            #pragma unroll
            for (int i = 0; i < 16; i++) {
                slm_dst[i] = a_src[i];
            }
            buf = next_buf;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    #undef SLM_A_STRIDE

    // === Store results ===
    // Each subgroup writes 32 rows x 16 cols
    // acc0..acc3: rows [0-7], [8-15], [16-23], [24-31], column = baseN + sg_lid
    __global half* c_base = C + baseM * N + baseN + sg_lid;

    #pragma unroll
    for (int r = 0; r < 8; r++) {
        c_base[(0 + r) * N] = convert_half(((float*)&acc0)[r]);
    }
    #pragma unroll
    for (int r = 0; r < 8; r++) {
        c_base[(8 + r) * N] = convert_half(((float*)&acc1)[r]);
    }
    #pragma unroll
    for (int r = 0; r < 8; r++) {
        c_base[(16 + r) * N] = convert_half(((float*)&acc2)[r]);
    }
    #pragma unroll
    for (int r = 0; r < 8; r++) {
        c_base[(24 + r) * N] = convert_half(((float*)&acc3)[r]);
    }
}
```

## Best implementation so far:

This is the code of the best kernel we have tested so far (Result: Correctness score: 5 / 5 (compiles and correct), runtime in ms: 1.120):
```OCL
// GEMM: C[M,N] = A[M,K] x B[K,N], row-major, FP16 in/out, FP32 accumulation
// Architecture: 64 WIs (4 SGs of 16), TILE 32x64x32, A in SLM (double-buffered), B from global
// Launch: GWS = (N/64 * 64, M/32), LWS = (64, 1)
// Each WG computes a 32(M) x 64(N) output tile
// 4 subgroups, each handles 32 rows x 16 cols
// K-loop steps by 32, with double-buffered A in SLM

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
    const int wg_n = get_group_id(0);  // which 64-col tile
    const int wg_m = get_group_id(1);  // which 32-row tile

    const int lid = get_local_id(0);   // 0..63
    const int sg_id = lid / 16;        // subgroup index 0..3
    const int sg_lid = get_sub_group_local_id(); // 0..15

    const int baseM = wg_m * 32;
    const int baseN = wg_n * 64 + sg_id * 16;

    // SLM for A: double-buffered, 32 rows x 32 cols
    // Stride = 32 halfs (no padding needed since 32*2=64 bytes is aligned)
    // Actually use stride=32 for simplicity; bank conflicts minimal with 16-wide access
    #define A_SLM_STRIDE 32
    #define A_SLM_SIZE (32 * A_SLM_STRIDE)
    __local half slm_A[2 * A_SLM_SIZE];

    // Accumulators: 32 rows x 16 cols = 4 x float8
    float8 acc0 = (float8)(0.0f);
    float8 acc1 = (float8)(0.0f);
    float8 acc2 = (float8)(0.0f);
    float8 acc3 = (float8)(0.0f);

    // Cooperative A load mapping:
    // 64 WIs load 32x32 = 1024 halfs = 16 halfs per WI
    // Each WI loads one half-row: lid/2 = row (0..31), (lid%2)*16 = col offset (0 or 16)
    const int a_load_row = lid / 2;         // 0..31
    const int a_load_col = (lid & 1) * 16;  // 0 or 16

    // Preload first A tile (k=0..31) into SLM buffer 0
    {
        __global const half* a_src = A + (baseM + a_load_row) * K + a_load_col;
        __local half* a_dst = slm_A + a_load_row * A_SLM_STRIDE + a_load_col;
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            a_dst[i] = a_src[i];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const int k_iters = K / 32;  // K=2048 -> 64 iterations
    int cur_buf = 0;

    for (int ki = 0; ki < k_iters; ki++) {
        const int k_base = ki * 32;

        // Pointer to current A buffer in SLM
        __local const half* a_slm = slm_A + cur_buf * A_SLM_SIZE;

        // Load B and compute for k_base..k_base+15 (first k16 step)
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

        // Load A from SLM for first k16: rows 0-7, 8-15, 16-23, 24-31
        short8 a00, a01, a02, a03;
        {
            __local const half* a_k0 = a_slm + sg_lid; // k-column = sg_lid
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

        // DPAS for first k16
        acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a00, b0, acc0);
        acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a01, b0, acc1);
        acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a02, b0, acc2);
        acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a03, b0, acc3);

        // Load B for second k16 (k_base+16..k_base+31)
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

        // Load A from SLM for second k16 (offset by 16 in k-dimension)
        short8 a10, a11, a12, a13;
        {
            __local const half* a_k1 = a_slm + 16 + sg_lid; // k-column = 16 + sg_lid
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

        // DPAS for second k16
        acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a10, b1, acc0);
        acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a11, b1, acc1);
        acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a12, b1, acc2);
        acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a13, b1, acc3);

        // Load next A tile into the other SLM buffer (double buffering)
        if (ki < k_iters - 1) {
            int next_buf = 1 - cur_buf;
            int next_k = k_base + 32;
            __global const half* a_src = A + (baseM + a_load_row) * K + next_k + a_load_col;
            __local half* a_dst = slm_A + next_buf * A_SLM_SIZE + a_load_row * A_SLM_STRIDE + a_load_col;
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                a_dst[i] = a_src[i];
            }
            cur_buf = next_buf;
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store results: each subgroup writes 32 rows x 16 cols
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
}
```

### Previous kernel and evaluation:
Here is the last OCL kernel we tested (Result: Correctness score: 0 / 5 (compilation error), runtime in ms: -1.000):
```OCL
// Launch: GWS = ((N/64)*16, (M/32)*4, 1), LWS = (16, 4, 1)
// Subgroup size = 16
// TILE: 32x64x32, A in SLM (double-buffered), B from global/L2
// DPAS: intel_sub_group_f16_f16_matrix_mad_k16(short8 a, int8 b, float8 acc)
//   Computes 8 rows x 16K x 16 cols per call
//   a = short8 (8 rows, each short = 2 half values packed, but for k16: 8 rows × 16 halfs)
//   b = int8 per WI (16K × 1 col, 16 WIs = 16 cols)
//   acc/result = float8 (8 rows for this WI's column)

#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable

#define TILE_M 32
#define TILE_N 64
#define TILE_K 32
#define SG_SIZE 16
#define NUM_SG 4
// SLM stride with padding to reduce bank conflicts
#define SLM_STRIDE 34

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(16, 4, 1)))
__kernel void matmul(
    __global const half* A,
    __global const half* B,
    __global half* C,
    const int M,
    const int K,
    const int N)
{
    // Work-group and subgroup identification
    const int wg_n = get_group_id(0);  // which 64-col tile
    const int wg_m = get_group_id(1);  // which 32-row tile
    const int sg_id = get_local_id(1); // subgroup index 0..3
    const int lane = get_local_id(0);  // lane within subgroup 0..15

    // Global tile offsets
    const int tile_row = wg_m * TILE_M;
    const int tile_col = wg_n * TILE_N;

    // Each subgroup handles 16 columns
    const int sg_col_offset = tile_col + sg_id * SG_SIZE;

    // SLM for A: double-buffered, 32 rows × 32 K elements with padding
    __local half slm_A[2 * TILE_M * SLM_STRIDE];

    // Accumulators: each subgroup computes 32 rows × 16 cols
    // 4 DPAS calls of 8 rows each = 32 rows
    float8 acc0 = (float8)(0.0f);
    float8 acc1 = (float8)(0.0f);
    float8 acc2 = (float8)(0.0f);
    float8 acc3 = (float8)(0.0f);

    // Cooperative A load: 64 WIs load 32×32 = 1024 halfs
    // Each WI loads 1024/64 = 16 halfs
    const int flat_id = sg_id * SG_SIZE + lane; // 0..63
    // Each WI loads 16 consecutive halfs
    // Map: flat_id covers 32 rows × 32 cols = 1024 elements
    // 16 elements per WI, so WI i loads elements [i*16 .. i*16+15]
    // Row = (flat_id * 16) / 32, starting col = (flat_id * 16) % 32
    // With 64 WIs × 16 = 1024 elements = 32×32 ✓
    // flat_id * 16: row = flat_id / 2, col_start = (flat_id % 2) * 16
    const int a_load_row = flat_id / 2;  // 0..31
    const int a_load_col_base = (flat_id % 2) * 16; // 0 or 16

    // Prefill first SLM buffer (buffer 0)
    {
        __global const half* a_src = A + (tile_row + a_load_row) * K + a_load_col_base;
        __local half* a_dst = slm_A + a_load_row * SLM_STRIDE + a_load_col_base;
        // Load 16 halfs
        half8 v0 = vload8(0, a_src);
        half8 v1 = vload8(1, a_src);
        vstore8(v0, 0, a_dst);
        vstore8(v1, 0, a_dst + 8);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int buf = 0;

    // Main K-loop: K/TILE_K iterations, unrolled 2x
    // Since K is divisible by 32, we have K/32 iterations
    const int k_iters = K / TILE_K;

    for (int ki = 0; ki < k_iters - 1; ki++) {
        int k_offset = ki * TILE_K;
        int next_k_offset = (ki + 1) * TILE_K;

        // Start loading next A tile into other buffer
        int next_buf = 1 - buf;
        {
            __global const half* a_src = A + (tile_row + a_load_row) * K + next_k_offset + a_load_col_base;
            __local half* a_dst = slm_A + next_buf * (TILE_M * SLM_STRIDE) + a_load_row * SLM_STRIDE + a_load_col_base;
            half8 v0 = vload8(0, a_src);
            half8 v1 = vload8(1, a_src);
            vstore8(v0, 0, a_dst);
            vstore8(v1, 0, a_dst + 8);
        }

        // Compute with current buffer
        // Process TILE_K=32 in two steps of 16 (DPAS k16)
        __local half* a_base = slm_A + buf * (TILE_M * SLM_STRIDE);

        #pragma unroll
        for (int kk = 0; kk < 2; kk++) {
            int k_inner = kk * 16;

            // Load B: each subgroup needs 16K × 16N from global
            // B is at row (k_offset + k_inner), col sg_col_offset
            // Each WI in the SG holds one column, needs 16 K values
            // int8 b means 8 ints = 16 halfs packed as int (2 halfs per int)
            __global const half* b_ptr = B + (k_offset + k_inner) * N + sg_col_offset + lane;
            int8 b_val;
            // Load 16 rows of B, each row stride = N
            half b_raw[16];
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                b_raw[i] = b_ptr[i * N];
            }
            // Pack into int8 (2 halfs per int)
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                ushort lo = as_ushort(b_raw[2*i]);
                ushort hi = as_ushort(b_raw[2*i+1]);
                ((int*)&b_val)[i] = (int)lo | ((int)hi << 16);
            }

            // Load A from SLM and compute DPAS for 4 blocks of 8 rows
            #pragma unroll
            for (int rm = 0; rm < 4; rm++) {
                // Load 8 rows × 16 K from SLM
                // Each row has SLM_STRIDE halfs, we need 16 consecutive
                short8 a_val;
                __local half* a_row_base = a_base + (rm * 8) * SLM_STRIDE + k_inner;

                // Load 8 rows × 16 halfs = 128 halfs
                // For DPAS a operand: short8 where each short contains 2 halfs
                // Actually for k16: a is short8 per WI, representing 8 rows
                // Each WI lane contributes to the systolic operation
                // The a operand is replicated across the subgroup
                // short8 a: element i = row i's k-pair for this lane's position

                // For intel_sub_group_f16_f16_matrix_mad_k16:
                // a = short8: 8 rows, each short = 2 consecutive f16 values at k positions [2*lane, 2*lane+1]
                #pragma unroll
                for (int r = 0; r < 8; r++) {
                    __local half* row_ptr = a_row_base + r * SLM_STRIDE;
                    ushort lo = as_ushort(row_ptr[2 * lane]);
                    ushort hi = as_ushort(row_ptr[2 * lane + 1]);
                    ((short*)&a_val)[r] = as_short((ushort)lo | ((uint)hi << 16));
                }
                // Actually short holds 2 bytes = 1 half. Let me reconsider.
                // short8 a: for k16 on exec_size=16:
                // Each lane holds: for each of 8 rows, a pair of halfs at k=[2*lane, 2*lane+1]
                // Packed as short (16-bit) = one half? No...
                // 
                // Let me reconsider the DPAS intrinsic format:
                // intel_sub_group_f16_f16_matrix_mad_k16(short8 a, int8 b, float8 acc)
                // With exec_size=16 and k=16:
                // a: short8 per WI. 8 = repcount (rows). Each short = 1 pair? 
                // Actually: k16 means depth=16. With f16, depth in terms of 32-bit = 8.
                // So the intrinsic name says k16 but internally it's depth=8 with 2 f16 per 32-bit slot.
                // a: short8 → 8 shorts = 16 bytes = 8 f16 values? No, that's only 8 halfs.
                // 
                // I think the correct interpretation:
                // For repcount=8, depth=8 (k=16 for f16):
                // a: each element of short8 corresponds to one row, containing the k-slice for that row
                //    packed as: 2 halfs in one short? No, short is 16 bits = 1 half.
                //
                // Let me look at this differently. The DPAS for f16 with k16:
                // a operand: 8 rows × 16 k-elements. Total = 128 halfs = 256 bytes.
                // Distributed across 16 lanes: 256/16 = 16 bytes per lane = short8 (8 × 2 bytes). ✓
                // So each lane holds 16 bytes of A data = 8 shorts = 8 halfs? 
                // No: 128 halfs / 16 lanes = 8 halfs per lane = 16 bytes = short8. ✓
                // Layout: lane l holds halfs at k-positions that map to lane l.
                // For VNNI/crosspack=2: pairs of k are packed together.
                // Lane l holds: row0_k[2l,2l+1], row1_k[2l,2l+1], ..., row7_k[2l,2l+1]? 
                // That would be 8 pairs = 16 halfs = 32 bytes ≠ short8.
                //
                // Actually wait: 8 rows × 16 k = 128 halfs. 128 halfs / 16 lanes = 8 halfs/lane.
                // 8 halfs = 16 bytes. short8 = 8 × 2 = 16 bytes. ✓
                // So each lane gets 8 halfs. The mapping:
                // With crosspack=2, lane l gets: for each row r (0..7), one half at k=l (or similar).
                // Actually no. Let me think about it as:
                // 8 rows × 16k. Crosspack=2 means pairs in k. So we have 8 rows × 8 pairs.
                // Distributed across 16 lanes... hmm.
                //
                // I think the correct layout for A in DPAS k16 f16 with exec_size=16:
                // The A matrix is 8×16 (rows×k). It's stored in VNNI format:
                // Logical: A[row][k], row=0..7, k=0..15
                // Physical in register: short8 per lane.
                // Lane l, element r: A[r][l] as a half stored in short.
                // So each lane l holds column l of the 8×16 A matrix? 
                // But that's only 8 halfs for 16 k-values... doesn't work for k=16.
                //
                // OK let me just use the practical approach:
                // For k16 with f16, the A operand should be ushort8 where:
                // Lane l, element i: contains 2 packed halfs A[i/1][...] 
                // 
                // Actually, I'll use a simpler model. Looking at real code:
                // For intel_sub_group_f16_f16_matrix_mad_k16 with exec_size=16:
                // a = short8: this is 8 rows × (k16 / 16 lanes) × 2 halfs_per_short... 
                //
                // I think the answer is:
                // short8 a: 8 elements, each is a short (16 bits = 1 half).
                // But we need k=16 and 8 rows. 8×16 = 128 halfs, /16 lanes = 8 halfs/lane.
                // So short8 = 8 halfs per lane. Each lane l holds one k-element per row:
                // a[r] = A[r][l] (as half, stored in short)
                // But that only covers k=16 if we have 16 lanes each holding a different k.
                // Yes! Lane l holds k=l for all 8 rows. So a[r] = as_short(A[r][l]).
                // This makes sense for exec_size=16, k=16.

                #pragma unroll
                for (int r = 0; r < 8; r++) {
                    __local half* row_ptr = a_row_base + r * SLM_STRIDE;
                    ((short*)&a_val)[r] = as_short(row_ptr[lane]);
                }

                // Compute DPAS
                float8* acc_ptr;
                if (rm == 0) acc_ptr = &acc0;
                else if (rm == 1) acc_ptr = &acc1;
                else if (rm == 2) acc_ptr = &acc2;
                else acc_ptr = &acc3;

                *acc_ptr = intel_sub_group_f16_f16_matrix_mad_k16(a_val, b_val, *acc_ptr);
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
        buf = next_buf;
    }

    // Last iteration (no next load needed)
    {
        int k_offset = (k_iters - 1) * TILE_K;
        __local half* a_base = slm_A + buf * (TILE_M * SLM_STRIDE);

        #pragma unroll
        for (int kk = 0; kk < 2; kk++) {
            int k_inner = kk * 16;

            __global const half* b_ptr = B + (k_offset + k_inner) * N + sg_col_offset + lane;
            int8 b_val;
            half b_raw[16];
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                b_raw[i] = b_ptr[i * N];
            }
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                ushort lo = as_ushort(b_raw[2*i]);
                ushort hi = as_ushort(b_raw[2*i+1]);
                ((int*)&b_val)[i] = (int)lo | ((int)hi << 16);
            }

            #pragma unroll
            for (int rm = 0; rm < 4; rm++) {
                short8 a_val;
                __local half* a_row_base = a_base + (rm * 8) * SLM_STRIDE + k_inner;

                #pragma unroll
                for (int r = 0; r < 8; r++) {
                    __local half* row_ptr = a_row_base + r * SLM_STRIDE;
                    ((short*)&a_val)[r] = as_short(row_ptr[lane]);
                }

                float8* acc_ptr;
                if (rm == 0) acc_ptr = &acc0;
                else if (rm == 1) acc_ptr = &acc1;
                else if (rm == 2) acc_ptr = &acc2;
                else acc_ptr = &acc3;

                *acc_ptr = intel_sub_group_f16_f16_matrix_mad_k16(a_val, b_val, *acc_ptr);
            }
        }
    }

    // Store results
    // Each subgroup writes 32 rows × 16 cols
    // acc0..acc3: each float8 = 8 rows, lane = column within the 16-col subgroup tile
    __global half* c_base = C + tile_row * N + sg_col_offset + lane;

    #pragma unroll
    for (int r = 0; r < 8; r++) {
        c_base[(0*8 + r) * N] = convert_half(((float*)&acc0)[r]);
    }
    #pragma unroll
    for (int r = 0; r < 8; r++) {
        c_base[(1*8 + r) * N] = convert_half(((float*)&acc1)[r]);
    }
    #pragma unroll
    for (int r = 0; r < 8; r++) {
        c_base[(2*8 + r) * N] = convert_half(((float*)&acc2)[r]);
    }
    #pragma unroll
    for (int r = 0; r < 8; r++) {
        c_base[(3*8 + r) * N] = convert_half(((float*)&acc3)[r]);
    }
}
```

Console output from running this kernel:
Syntax precheck failed: clBuildProgram failed: BUILD_PROGRAM_FAILURE - clBuildProgram failed: BUILD_PROGRAM_FAILURE - clBuildProgram failed: BUILD_PROGRAM_FAILURE

Build on <pyopencl.Device 'Intel(R) Arc(TM) B580 Graphics' on 'Intel(R) OpenCL Graphics' at 0x5b1b7e7feb90>:

4:12:26: warning: OpenCL extension 'cl_intel_requi

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

- **Vectorized Memory Access**: Replace scalar loads with SYCL vectors (`float4` for general use, `float8`/`float16` for bulk transfers, `float2` for tight register pressure). Ensure adjacent work-items access adjacent memory addresses for coalesced access.
- **Work-Group Reductions**: Replace atomic operations with O(log N) tree-based reductions in local memory. Synchronize with `group_barrier()` between iterations.