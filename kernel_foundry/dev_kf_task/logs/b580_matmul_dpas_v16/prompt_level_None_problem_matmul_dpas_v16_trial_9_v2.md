

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

### Version 1 (Result: Correctness score: 5 / 5 (compiles and correct), runtime in ms: 1.460):
```OCL
// GEMM: C[M,N] = A[M,K] * B[K,N], all half, f32 accumulation
// Launch: GWS = (N/64 * 64, M/32), LWS = (64, 1)
// 4 subgroups of 16 WIs. Tile: 32x64x32. A in SLM (double-buffered), B from global.
// K=2048, M=2048, N=2048 - all divisible by 64, no remainder handling needed.

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
    #define TILE_M 32
    #define TILE_N 64
    #define TILE_K 32
    #define SLM_A_STRIDE 32

    const int lid = get_local_id(0);
    const int sg_id = get_sub_group_id();
    const int sg_lid = get_sub_group_local_id();

    const int wg_n = get_group_id(0);
    const int wg_m = get_group_id(1);

    const int base_row = wg_m * TILE_M;
    const int base_col = wg_n * TILE_N;
    const int sg_col_offset = sg_id * 16;

    __local half slm_A[2 * TILE_M * SLM_A_STRIDE];

    float8 acc0 = 0.0f;
    float8 acc1 = 0.0f;
    float8 acc2 = 0.0f;
    float8 acc3 = 0.0f;

    // Cooperative A load: 64 WIs load 32x32 = 1024 halfs, 16 per WI
    const int a_row = lid / 2;
    const int a_col_base = (lid & 1) * 16;

    // Prefill first A tile into SLM buffer 0
    {
        __global const half* a_ptr = A + (base_row + a_row) * K + a_col_base;
        __local half* slm_dst = slm_A + a_row * SLM_A_STRIDE + a_col_base;
        half16 a_val = vload16(0, a_ptr);
        vstore16(a_val, 0, slm_dst);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int buf = 0;

    for (int k = 0; k < K; k += TILE_K) {
        const int slm_base = buf * (TILE_M * SLM_A_STRIDE);

        // Load B for k-step 0 (k..k+15)
        int8 b0;
        {
            __global const half* b_ptr = B + k * N + base_col + sg_col_offset + sg_lid;
            b0.s0 = as_int((half2)(b_ptr[0*N], b_ptr[1*N]));
            b0.s1 = as_int((half2)(b_ptr[2*N], b_ptr[3*N]));
            b0.s2 = as_int((half2)(b_ptr[4*N], b_ptr[5*N]));
            b0.s3 = as_int((half2)(b_ptr[6*N], b_ptr[7*N]));
            b0.s4 = as_int((half2)(b_ptr[8*N], b_ptr[9*N]));
            b0.s5 = as_int((half2)(b_ptr[10*N], b_ptr[11*N]));
            b0.s6 = as_int((half2)(b_ptr[12*N], b_ptr[13*N]));
            b0.s7 = as_int((half2)(b_ptr[14*N], b_ptr[15*N]));
        }

        // Load A from SLM for k-step 0 (cols 0..15)
        short8 a00, a01, a02, a03;
        {
            __local half* abase = slm_A + slm_base + sg_lid;
            a00.s0 = as_short(abase[0*SLM_A_STRIDE]); a00.s1 = as_short(abase[1*SLM_A_STRIDE]);
            a00.s2 = as_short(abase[2*SLM_A_STRIDE]); a00.s3 = as_short(abase[3*SLM_A_STRIDE]);
            a00.s4 = as_short(abase[4*SLM_A_STRIDE]); a00.s5 = as_short(abase[5*SLM_A_STRIDE]);
            a00.s6 = as_short(abase[6*SLM_A_STRIDE]); a00.s7 = as_short(abase[7*SLM_A_STRIDE]);

            abase += 8*SLM_A_STRIDE;
            a01.s0 = as_short(abase[0*SLM_A_STRIDE]); a01.s1 = as_short(abase[1*SLM_A_STRIDE]);
            a01.s2 = as_short(abase[2*SLM_A_STRIDE]); a01.s3 = as_short(abase[3*SLM_A_STRIDE]);
            a01.s4 = as_short(abase[4*SLM_A_STRIDE]); a01.s5 = as_short(abase[5*SLM_A_STRIDE]);
            a01.s6 = as_short(abase[6*SLM_A_STRIDE]); a01.s7 = as_short(abase[7*SLM_A_STRIDE]);

            abase += 8*SLM_A_STRIDE;
            a02.s0 = as_short(abase[0*SLM_A_STRIDE]); a02.s1 = as_short(abase[1*SLM_A_STRIDE]);
            a02.s2 = as_short(abase[2*SLM_A_STRIDE]); a02.s3 = as_short(abase[3*SLM_A_STRIDE]);
            a02.s4 = as_short(abase[4*SLM_A_STRIDE]); a02.s5 = as_short(abase[5*SLM_A_STRIDE]);
            a02.s6 = as_short(abase[6*SLM_A_STRIDE]); a02.s7 = as_short(abase[7*SLM_A_STRIDE]);

            abase += 8*SLM_A_STRIDE;
            a03.s0 = as_short(abase[0*SLM_A_STRIDE]); a03.s1 = as_short(abase[1*SLM_A_STRIDE]);
            a03.s2 = as_short(abase[2*SLM_A_STRIDE]); a03.s3 = as_short(abase[3*SLM_A_STRIDE]);
            a03.s4 = as_short(abase[4*SLM_A_STRIDE]); a03.s5 = as_short(abase[5*SLM_A_STRIDE]);
            a03.s6 = as_short(abase[6*SLM_A_STRIDE]); a03.s7 = as_short(abase[7*SLM_A_STRIDE]);
        }

        // DPAS k-step 0
        acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a00, b0, acc0);
        acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a01, b0, acc1);
        acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a02, b0, acc2);
        acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a03, b0, acc3);

        // Load B for k-step 1 (k+16..k+31)
        int8 b1;
        {
            __global const half* b_ptr = B + (k+16) * N + base_col + sg_col_offset + sg_lid;
            b1.s0 = as_int((half2)(b_ptr[0*N], b_ptr[1*N]));
            b1.s1 = as_int((half2)(b_ptr[2*N], b_ptr[3*N]));
            b1.s2 = as_int((half2)(b_ptr[4*N], b_ptr[5*N]));
            b1.s3 = as_int((half2)(b_ptr[6*N], b_ptr[7*N]));
            b1.s4 = as_int((half2)(b_ptr[8*N], b_ptr[9*N]));
            b1.s5 = as_int((half2)(b_ptr[10*N], b_ptr[11*N]));
            b1.s6 = as_int((half2)(b_ptr[12*N], b_ptr[13*N]));
            b1.s7 = as_int((half2)(b_ptr[14*N], b_ptr[15*N]));
        }

        // Load A from SLM for k-step 1 (cols 16..31)
        short8 a10, a11, a12, a13;
        {
            __local half* abase = slm_A + slm_base + 16 + sg_lid;
            a10.s0 = as_short(abase[0*SLM_A_STRIDE]); a10.s1 = as_short(abase[1*SLM_A_STRIDE]);
            a10.s2 = as_short(abase[2*SLM_A_STRIDE]); a10.s3 = as_short(abase[3*SLM_A_STRIDE]);
            a10.s4 = as_short(abase[4*SLM_A_STRIDE]); a10.s5 = as_short(abase[5*SLM_A_STRIDE]);
            a10.s6 = as_short(abase[6*SLM_A_STRIDE]); a10.s7 = as_short(abase[7*SLM_A_STRIDE]);

            abase += 8*SLM_A_STRIDE;
            a11.s0 = as_short(abase[0*SLM_A_STRIDE]); a11.s1 = as_short(abase[1*SLM_A_STRIDE]);
            a11.s2 = as_short(abase[2*SLM_A_STRIDE]); a11.s3 = as_short(abase[3*SLM_A_STRIDE]);
            a11.s4 = as_short(abase[4*SLM_A_STRIDE]); a11.s5 = as_short(abase[5*SLM_A_STRIDE]);
            a11.s6 = as_short(abase[6*SLM_A_STRIDE]); a11.s7 = as_short(abase[7*SLM_A_STRIDE]);

            abase += 8*SLM_A_STRIDE;
            a12.s0 = as_short(abase[0*SLM_A_STRIDE]); a12.s1 = as_short(abase[1*SLM_A_STRIDE]);
            a12.s2 = as_short(abase[2*SLM_A_STRIDE]); a12.s3 = as_short(abase[3*SLM_A_STRIDE]);
            a12.s4 = as_short(abase[4*SLM_A_STRIDE]); a12.s5 = as_short(abase[5*SLM_A_STRIDE]);
            a12.s6 = as_short(abase[6*SLM_A_STRIDE]); a12.s7 = as_short(abase[7*SLM_A_STRIDE]);

            abase += 8*SLM_A_STRIDE;
            a13.s0 = as_short(abase[0*SLM_A_STRIDE]); a13.s1 = as_short(abase[1*SLM_A_STRIDE]);
            a13.s2 = as_short(abase[2*SLM_A_STRIDE]); a13.s3 = as_short(abase[3*SLM_A_STRIDE]);
            a13.s4 = as_short(abase[4*SLM_A_STRIDE]); a13.s5 = as_short(abase[5*SLM_A_STRIDE]);
            a13.s6 = as_short(abase[6*SLM_A_STRIDE]); a13.s7 = as_short(abase[7*SLM_A_STRIDE]);
        }

        // Prefetch next A tile into other buffer
        int next_k = k + TILE_K;
        if (next_k < K) {
            int next_buf = 1 - buf;
            __global const half* a_ptr = A + (base_row + a_row) * K + next_k + a_col_base;
            __local half* slm_dst = slm_A + next_buf * (TILE_M * SLM_A_STRIDE) + a_row * SLM_A_STRIDE + a_col_base;
            half16 a_val = vload16(0, a_ptr);
            vstore16(a_val, 0, slm_dst);
        }

        // DPAS k-step 1
        acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a10, b1, acc0);
        acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a11, b1, acc1);
        acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a12, b1, acc2);
        acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a13, b1, acc3);

        buf = 1 - buf;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write C
    const int out_col = base_col + sg_col_offset + sg_lid;
    __global half* c_ptr = C + base_row * N + out_col;

    #pragma unroll
    for (int r = 0; r < 8; r++) c_ptr[r * N] = convert_half(acc0[r]);
    c_ptr += 8 * N;
    #pragma unroll
    for (int r = 0; r < 8; r++) c_ptr[r * N] = convert_half(acc1[r]);
    c_ptr += 8 * N;
    #pragma unroll
    for (int r = 0; r < 8; r++) c_ptr[r * N] = convert_half(acc2[r]);
    c_ptr += 8 * N;
    #pragma unroll
    for (int r = 0; r < 8; r++) c_ptr[r * N] = convert_half(acc3[r]);
}
```

## Best implementation so far:

This is the code of the best kernel we have tested so far (Result: Correctness score: 5 / 5 (compiles and correct), runtime in ms: 1.300):
```OCL
// GEMM: C[M,N] = A[M,K] * B[K,N], all half, f32 accumulation
// Launch: GWS = (N/64 * 64, M/32), LWS = (64, 1)
// 4 subgroups of 16 WIs. Tile: 32x64x32. A in SLM (double-buffered), B from global.
// K=2048, M=2048, N=2048 - all divisible by 64.

#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable

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
    const int lid = get_local_id(0);
    const int sg_id = get_sub_group_id();
    const int sg_lid = get_sub_group_local_id();

    const int wg_m = get_group_id(1);
    const int wg_n = get_group_id(0);

    const int base_row = wg_m * 32;
    const int base_col = wg_n * 64;
    const int sg_col = base_col + sg_id * 16;

    // SLM double buffer for A: 2 * 32 rows * 32 cols = 4096 bytes
    #define SLM_STRIDE 32
    __local half slm_A[2 * 32 * SLM_STRIDE];

    // Accumulators
    float8 acc0 = 0.0f;
    float8 acc1 = 0.0f;
    float8 acc2 = 0.0f;
    float8 acc3 = 0.0f;

    // Cooperative A load: 64 WIs, 32 rows x 32 cols = 1024 halfs, 16 per WI
    const int a_row = lid >> 1;           // 0..31
    const int a_col_base = (lid & 1) << 4; // 0 or 16

    // Preload first A tile into buffer 0
    {
        __global const half* a_src = A + (base_row + a_row) * K + a_col_base;
        __local half* a_dst = slm_A + a_row * SLM_STRIDE + a_col_base;
        half8 v0 = vload8(0, a_src);
        half8 v1 = vload8(1, a_src);
        vstore8(v0, 0, a_dst);
        vstore8(v1, 0, a_dst + 8);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Main K-loop: step by 64 (2x unrolled), K=2048 divides by 64
    for (int k = 0; k < K; k += 64) {

        // ===== Phase 0: Compute from buffer 0, prefetch A(k+32) into buffer 1 =====
        {
            __local const half* slm_base = slm_A;
            __global const half* b_base = B + k * N + sg_col + sg_lid;

            // --- K-step 0: k..k+15 ---
            // Read A from SLM: each WI reads column sg_lid from 8 rows
            // For intel_sub_group_f16_f16_matrix_mad_k16: a is short8
            // Each WI w holds a[r] = A[row_base+r][k_base+w] as short (one f16 reinterpreted)
            short8 a00, a10, a20, a30;
            {
                __local const half* ap = slm_base + sg_lid;
                a00.s0 = as_short(ap[0*SLM_STRIDE]); a00.s1 = as_short(ap[1*SLM_STRIDE]);
                a00.s2 = as_short(ap[2*SLM_STRIDE]); a00.s3 = as_short(ap[3*SLM_STRIDE]);
                a00.s4 = as_short(ap[4*SLM_STRIDE]); a00.s5 = as_short(ap[5*SLM_STRIDE]);
                a00.s6 = as_short(ap[6*SLM_STRIDE]); a00.s7 = as_short(ap[7*SLM_STRIDE]);
                ap += 8*SLM_STRIDE;
                a10.s0 = as_short(ap[0*SLM_STRIDE]); a10.s1 = as_short(ap[1*SLM_STRIDE]);
                a10.s2 = as_short(ap[2*SLM_STRIDE]); a10.s3 = as_short(ap[3*SLM_STRIDE]);
                a10.s4 = as_short(ap[4*SLM_STRIDE]); a10.s5 = as_short(ap[5*SLM_STRIDE]);
                a10.s6 = as_short(ap[6*SLM_STRIDE]); a10.s7 = as_short(ap[7*SLM_STRIDE]);
                ap += 8*SLM_STRIDE;
                a20.s0 = as_short(ap[0*SLM_STRIDE]); a20.s1 = as_short(ap[1*SLM_STRIDE]);
                a20.s2 = as_short(ap[2*SLM_STRIDE]); a20.s3 = as_short(ap[3*SLM_STRIDE]);
                a20.s4 = as_short(ap[4*SLM_STRIDE]); a20.s5 = as_short(ap[5*SLM_STRIDE]);
                a20.s6 = as_short(ap[6*SLM_STRIDE]); a20.s7 = as_short(ap[7*SLM_STRIDE]);
                ap += 8*SLM_STRIDE;
                a30.s0 = as_short(ap[0*SLM_STRIDE]); a30.s1 = as_short(ap[1*SLM_STRIDE]);
                a30.s2 = as_short(ap[2*SLM_STRIDE]); a30.s3 = as_short(ap[3*SLM_STRIDE]);
                a30.s4 = as_short(ap[4*SLM_STRIDE]); a30.s5 = as_short(ap[5*SLM_STRIDE]);
                a30.s6 = as_short(ap[6*SLM_STRIDE]); a30.s7 = as_short(ap[7*SLM_STRIDE]);
            }

            // Read B: 16 k-rows, each WI reads its column, pack pairs into int
            int8 b0;
            {
                __global const half* bp = b_base;
                b0.s0 = as_int((half2)(bp[0], bp[N]));
                b0.s1 = as_int((half2)(bp[2*N], bp[3*N]));
                b0.s2 = as_int((half2)(bp[4*N], bp[5*N]));
                b0.s3 = as_int((half2)(bp[6*N], bp[7*N]));
                b0.s4 = as_int((half2)(bp[8*N], bp[9*N]));
                b0.s5 = as_int((half2)(bp[10*N], bp[11*N]));
                b0.s6 = as_int((half2)(bp[12*N], bp[13*N]));
                b0.s7 = as_int((half2)(bp[14*N], bp[15*N]));
            }

            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a00, b0, acc0);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a10, b0, acc1);
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a20, b0, acc2);
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a30, b0, acc3);

            // --- K-step 1: k+16..k+31 ---
            short8 a01, a11, a21, a31;
            {
                __local const half* ap = slm_base + 16 + sg_lid;
                a01.s0 = as_short(ap[0*SLM_STRIDE]); a01.s1 = as_short(ap[1*SLM_STRIDE]);
                a01.s2 = as_short(ap[2*SLM_STRIDE]); a01.s3 = as_short(ap[3*SLM_STRIDE]);
                a01.s4 = as_short(ap[4*SLM_STRIDE]); a01.s5 = as_short(ap[5*SLM_STRIDE]);
                a01.s6 = as_short(ap[6*SLM_STRIDE]); a01.s7 = as_short(ap[7*SLM_STRIDE]);
                ap += 8*SLM_STRIDE;
                a11.s0 = as_short(ap[0*SLM_STRIDE]); a11.s1 = as_short(ap[1*SLM_STRIDE]);
                a11.s2 = as_short(ap[2*SLM_STRIDE]); a11.s3 = as_short(ap[3*SLM_STRIDE]);
                a11.s4 = as_short(ap[4*SLM_STRIDE]); a11.s5 = as_short(ap[5*SLM_STRIDE]);
                a11.s6 = as_short(ap[6*SLM_STRIDE]); a11.s7 = as_short(ap[7*SLM_STRIDE]);
                ap += 8*SLM_STRIDE;
                a21.s0 = as_short(ap[0*SLM_STRIDE]); a21.s1 = as_short(ap[1*SLM_STRIDE]);
                a21.s2 = as_short(ap[2*SLM_STRIDE]); a21.s3 = as_short(ap[3*SLM_STRIDE]);
                a21.s4 = as_short(ap[4*SLM_STRIDE]); a21.s5 = as_short(ap[5*SLM_STRIDE]);
                a21.s6 = as_short(ap[6*SLM_STRIDE]); a21.s7 = as_short(ap[7*SLM_STRIDE]);
                ap += 8*SLM_STRIDE;
                a31.s0 = as_short(ap[0*SLM_STRIDE]); a31.s1 = as_short(ap[1*SLM_STRIDE]);
                a31.s2 = as_short(ap[2*SLM_STRIDE]); a31.s3 = as_short(ap[3*SLM_STRIDE]);
                a31.s4 = as_short(ap[4*SLM_STRIDE]); a31.s5 = as_short(ap[5*SLM_STRIDE]);
                a31.s6 = as_short(ap[6*SLM_STRIDE]); a31.s7 = as_short(ap[7*SLM_STRIDE]);
            }

            int8 b1;
            {
                __global const half* bp = b_base + 16*N;
                b1.s0 = as_int((half2)(bp[0], bp[N]));
                b1.s1 = as_int((half2)(bp[2*N], bp[3*N]));
                b1.s2 = as_int((half2)(bp[4*N], bp[5*N]));
                b1.s3 = as_int((half2)(bp[6*N], bp[7*N]));
                b1.s4 = as_int((half2)(bp[8*N], bp[9*N]));
                b1.s5 = as_int((half2)(bp[10*N], bp[11*N]));
                b1.s6 = as_int((half2)(bp[12*N], bp[13*N]));
                b1.s7 = as_int((half2)(bp[14*N], bp[15*N]));
            }

            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a01, b1, acc0);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a11, b1, acc1);
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a21, b1, acc2);
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a31, b1, acc3);

            // Load A(k+32) into buffer 1
            {
                __global const half* a_src = A + (base_row + a_row) * K + (k + 32) + a_col_base;
                __local half* a_dst = slm_A + 32*SLM_STRIDE + a_row * SLM_STRIDE + a_col_base;
                half8 v0 = vload8(0, a_src);
                half8 v1 = vload8(1, a_src);
                vstore8(v0, 0, a_dst);
                vstore8(v1, 0, a_dst + 8);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // ===== Phase 1: Compute from buffer 1, prefetch A(k+64) into buffer 0 =====
        {
            __local const half* slm_base = slm_A + 32*SLM_STRIDE;
            __global const half* b_base = B + (k+32) * N + sg_col + sg_lid;

            short8 a00, a10, a20, a30;
            {
                __local const half* ap = slm_base + sg_lid;
                a00.s0 = as_short(ap[0*SLM_STRIDE]); a00.s1 = as_short(ap[1*SLM_STRIDE]);
                a00.s2 = as_short(ap[2*SLM_STRIDE]); a00.s3 = as_short(ap[3*SLM_STRIDE]);
                a00.s4 = as_short(ap[4*SLM_STRIDE]); a00.s5 = as_short(ap[5*SLM_STRIDE]);
                a00.s6 = as_short(ap[6*SLM_STRIDE]); a00.s7 = as_short(ap[7*SLM_STRIDE]);
                ap += 8*SLM_STRIDE;
                a10.s0 = as_short(ap[0*SLM_STRIDE]); a10.s1 = as_short(ap[1*SLM_STRIDE]);
                a10.s2 = as_short(ap[2*SLM_STRIDE]); a10.s3 = as_short(ap[3*SLM_STRIDE]);
                a10.s4 = as_short(ap[4*SLM_STRIDE]); a10.s5 = as_short(ap[5*SLM_STRIDE]);
                a10.s6 = as_short(ap[6*SLM_STRIDE]); a10.s7 = as_short(ap[7*SLM_STRIDE]);
                ap += 8*SLM_STRIDE;
                a20.s0 = as_short(ap[0*SLM_STRIDE]); a20.s1 = as_short(ap[1*SLM_STRIDE]);
                a20.s2 = as_short(ap[2*SLM_STRIDE]); a20.s3 = as_short(ap[3*SLM_STRIDE]);
                a20.s4 = as_short(ap[4*SLM_STRIDE]); a20.s5 = as_short(ap[5*SLM_STRIDE]);
                a20.s6 = as_short(ap[6*SLM_STRIDE]); a20.s7 = as_short(ap[7*SLM_STRIDE]);
                ap += 8*SLM_STRIDE;
                a30.s0 = as_short(ap[0*SLM_STRIDE]); a30.s1 = as_short(ap[1*SLM_STRIDE]);
                a30.s2 = as_short(ap[2*SLM_STRIDE]); a30.s3 = as_short(ap[3*SLM_STRIDE]);
                a30.s4 = as_short(ap[4*SLM_STRIDE]); a30.s5 = as_short(ap[5*SLM_STRIDE]);
                a30.s6 = as_short(ap[6*SLM_STRIDE]); a30.s7 = as_short(ap[7*SLM_STRIDE]);
            }

            int8 b0;
            {
                __global const half* bp = b_base;
                b0.s0 = as_int((half2)(bp[0], bp[N]));
                b0.s1 = as_int((half2)(bp[2*N], bp[3*N]));
                b0.s2 = as_int((half2)(bp[4*N], bp[5*N]));
                b0.s3 = as_int((half2)(bp[6*N], bp[7*N]));
                b0.s4 = as_int((half2)(bp[8*N], bp[9*N]));
                b0.s5 = as_int((half2)(bp[10*N], bp[11*N]));
                b0.s6 = as_int((half2)(bp[12*N], bp[13*N]));
                b0.s7 = as_int((half2)(bp[14*N], bp[15*N]));
            }

            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a00, b0, acc0);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a10, b0, acc1);
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a20, b0, acc2);
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a30, b0, acc3);

            short8 a01, a11, a21, a31;
            {
                __local const half* ap = slm_base + 16 + sg_lid;
                a01.s0 = as_short(ap[0*SLM_STRIDE]); a01.s1 = as_short(ap[1*SLM_STRIDE]);
                a01.s2 = as_short(ap[2*SLM_STRIDE]); a01.s3 = as_short(ap[3*SLM_STRIDE]);
                a01.s4 = as_short(ap[4*SLM_STRIDE]); a01.s5 = as_short(ap[5*SLM_STRIDE]);
                a01.s6 = as_short(ap[6*SLM_STRIDE]); a01.s7 = as_short(ap[7*SLM_STRIDE]);
                ap += 8*SLM_STRIDE;
                a11.s0 = as_short(ap[0*SLM_STRIDE]); a11.s1 = as_short(ap[1*SLM_STRIDE]);
                a11.s2 = as_short(ap[2*SLM_STRIDE]); a11.s3 = as_short(ap[3*SLM_STRIDE]);
                a11.s4 = as_short(ap[4*SLM_STRIDE]); a11.s5 = as_short(ap[5*SLM_STRIDE]);
                a11.s6 = as_short(ap[6*SLM_STRIDE]); a11.s7 = as_short(ap[7*SLM_STRIDE]);
                ap += 8*SLM_STRIDE;
                a21.s0 = as_short(ap[0*SLM_STRIDE]); a21.s1 = as_short(ap[1*SLM_STRIDE]);
                a21.s2 = as_short(ap[2*SLM_STRIDE]); a21.s3 = as_short(ap[3*SLM_STRIDE]);
                a21.s4 = as_short(ap[4*SLM_STRIDE]); a21.s5 = as_short(ap[5*SLM_STRIDE]);
                a21.s6 = as_short(ap[6*SLM_STRIDE]); a21.s7 = as_short(ap[7*SLM_STRIDE]);
                ap += 8*SLM_STRIDE;
                a31.s0 = as_short(ap[0*SLM_STRIDE]); a31.s1 = as_short(ap[1*SLM_STRIDE]);
                a31.s2 = as_short(ap[2*SLM_STRIDE]); a31.s3 = as_short(ap[3*SLM_STRIDE]);
                a31.s4 = as_short(ap[4*SLM_STRIDE]); a31.s5 = as_short(ap[5*SLM_STRIDE]);
                a31.s6 = as_short(ap[6*SLM_STRIDE]); a31.s7 = as_short(ap[7*SLM_STRIDE]);
            }

            int8 b1;
            {
                __global const half* bp = b_base + 16*N;
                b1.s0 = as_int((half2)(bp[0], bp[N]));
                b1.s1 = as_int((half2)(bp[2*N], bp[3*N]));
                b1.s2 = as_int((half2)(bp[4*N], bp[5*N]));
                b1.s3 = as_int((half2)(bp[6*N], bp[7*N]));
                b1.s4 = as_int((half2)(bp[8*N], bp[9*N]));
                b1.s5 = as_int((half2)(bp[10*N], bp[11*N]));
                b1.s6 = as_int((half2)(bp[12*N], bp[13*N]));
                b1.s7 = as_int((half2)(bp[14*N], bp[15*N]));
            }

            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a01, b1, acc0);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a11, b1, acc1);
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a21, b1, acc2);
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a31, b1, acc3);

            // Load A(k+64) into buffer 0
            if (k + 64 < K) {
                __global const half* a_src = A + (base_row + a_row) * K + (k + 64) + a_col_base;
                __local half* a_dst = slm_A + a_row * SLM_STRIDE + a_col_base;
                half8 v0 = vload8(0, a_src);
                half8 v1 = vload8(1, a_src);
                vstore8(v0, 0, a_dst);
                vstore8(v1, 0, a_dst + 8);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write C: each WI writes 32 rows to its column
    __global half* c_ptr = C + base_row * N + sg_col + sg_lid;
    c_ptr[0*N] = convert_half(acc0.s0); c_ptr[1*N] = convert_half(acc0.s1);
    c_ptr[2*N] = convert_half(acc0.s2); c_ptr[3*N] = convert_half(acc0.s3);
    c_ptr[4*N] = convert_half(acc0.s4); c_ptr[5*N] = convert_half(acc0.s5);
    c_ptr[6*N] = convert_half(acc0.s6); c_ptr[7*N] = convert_half(acc0.s7);
    c_ptr[8*N] = convert_half(acc1.s0); c_ptr[9*N] = convert_half(acc1.s1);
    c_ptr[10*N] = convert_half(acc1.s2); c_ptr[11*N] = convert_half(acc1.s3);
    c_ptr[12*N] = convert_half(acc1.s4); c_ptr[13*N] = convert_half(acc1.s5);
    c_ptr[14*N] = convert_half(acc1.s6); c_ptr[15*N] = convert_half(acc1.s7);
    c_ptr[16*N] = convert_half(acc2.s0); c_ptr[17*N] = convert_half(acc2.s1);
    c_ptr[18*N] = convert_half(acc2.s2); c_ptr[19*N] = convert_half(acc2.s3);
    c_ptr[20*N] = convert_half(acc2.s4); c_ptr[21*N] = convert_half(acc2.s5);
    c_ptr[22*N] = convert_half(acc2.s6); c_ptr[23*N] = convert_half(acc2.s7);
    c_ptr[24*N] = convert_half(acc3.s0); c_ptr[25*N] = convert_half(acc3.s1);
    c_ptr[26*N] = convert_half(acc3.s2); c_ptr[27*N] = convert_half(acc3.s3);
    c_ptr[28*N] = convert_half(acc3.s4); c_ptr[29*N] = convert_half(acc3.s5);
    c_ptr[30*N] = convert_half(acc3.s6); c_ptr[31*N] = convert_half(acc3.s7);
}
```

### Previous kernel and evaluation:
Here is the last OCL kernel we tested (Result: Correctness score: 3 / 5 (compiled but incorrect), runtime in ms: -1.000):
```OCL
// GEMM: C[M,N] = A[M,K] * B[K,N], all half, f32 accumulation
// Launch: GWS = (N/64 * 64, M/32), LWS = (64, 1)
// 4 subgroups of 16 WIs. Tile: 32x64x32. A in SLM (double-buffered), B from global.
// K=2048, M=2048, N=2048 - all divisible by 64.
// Optimizations: 2x K-unroll, no remainder, interleaved load/compute, no branches in loop.

#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable

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
    const int lid = get_local_id(0);
    const int sg_id = get_sub_group_id();
    const int sg_lid = get_sub_group_local_id();

    const int wg_m = get_group_id(1);
    const int wg_n = get_group_id(0);

    const int base_row = wg_m * 32;
    const int base_col = wg_n * 64;
    const int sg_col = base_col + sg_id * 16 + sg_lid;

    // SLM double buffer for A: 2 * 32 rows * 32 cols = 2048 halfs = 4096 bytes
    #define SLM_S 32
    __local half slm_A[2 * 32 * SLM_S];

    // Accumulators: 4 groups of 8 rows = 32 rows total
    float8 acc0 = 0.0f;
    float8 acc1 = 0.0f;
    float8 acc2 = 0.0f;
    float8 acc3 = 0.0f;

    // Cooperative A load mapping: 64 WIs load 32x32 = 1024 halfs, 16 per WI
    const int a_row = lid >> 1;            // 0..31
    const int a_col_base = (lid & 1) << 4; // 0 or 16

    // Precompute A row pointer base
    const int a_row_offset = (base_row + a_row) * K;

    // Preload first A tile (k=0..31) into buffer 0
    {
        __global const half* a_src = A + a_row_offset + a_col_base;
        __local half* a_dst = slm_A + a_row * SLM_S + a_col_base;
        half8 v0 = vload8(0, a_src);
        half8 v1 = vload8(1, a_src);
        vstore8(v0, 0, a_dst);
        vstore8(v1, 0, a_dst + 8);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Preload second A tile (k=32..63) into buffer 1
    {
        __global const half* a_src = A + a_row_offset + 32 + a_col_base;
        __local half* a_dst = slm_A + 32 * SLM_S + a_row * SLM_S + a_col_base;
        half8 v0 = vload8(0, a_src);
        half8 v1 = vload8(1, a_src);
        vstore8(v0, 0, a_dst);
        vstore8(v1, 0, a_dst + 8);
    }

    // Main K-loop: step by 64 (2x unrolled), K=2048 divides evenly by 64
    // Total iterations: 2048/64 = 32
    // We preloaded buffers 0 and 1 for k=0. Each iteration:
    //   - Compute from buffer 0 (current k)
    //   - Compute from buffer 1 (current k+32)  
    //   - Load next buffer 0 (k+64) and buffer 1 (k+96)
    // Last iteration: k=1984, no next load needed.

    const int K_MAIN = K - 64; // = 1984, last iteration where we still need to prefetch

    for (int k = 0; k < K; k += 64) {
        barrier(CLK_LOCAL_MEM_FENCE);

        // ===== Phase 0: Compute from buffer 0 (k..k+31) =====
        {
            __local const half* slm_base = slm_A;
            __global const half* b_base = B + k * N + sg_col;

            // Load B k-step 0 (k..k+15) - start early for latency hiding
            int8 b0;
            {
                __global const half* bp = b_base;
                b0.s0 = as_int((half2)(bp[0], bp[N]));
                b0.s1 = as_int((half2)(bp[2*N], bp[3*N]));
                b0.s2 = as_int((half2)(bp[4*N], bp[5*N]));
                b0.s3 = as_int((half2)(bp[6*N], bp[7*N]));
                b0.s4 = as_int((half2)(bp[8*N], bp[9*N]));
                b0.s5 = as_int((half2)(bp[10*N], bp[11*N]));
                b0.s6 = as_int((half2)(bp[12*N], bp[13*N]));
                b0.s7 = as_int((half2)(bp[14*N], bp[15*N]));
            }

            // Load A from SLM k-step 0 (cols 0..15)
            short8 a00, a10, a20, a30;
            {
                __local const half* ap = slm_base + sg_lid;
                a00.s0 = as_short(ap[0*SLM_S]); a00.s1 = as_short(ap[1*SLM_S]);
                a00.s2 = as_short(ap[2*SLM_S]); a00.s3 = as_short(ap[3*SLM_S]);
                a00.s4 = as_short(ap[4*SLM_S]); a00.s5 = as_short(ap[5*SLM_S]);
                a00.s6 = as_short(ap[6*SLM_S]); a00.s7 = as_short(ap[7*SLM_S]);
                ap += 8*SLM_S;
                a10.s0 = as_short(ap[0*SLM_S]); a10.s1 = as_short(ap[1*SLM_S]);
                a10.s2 = as_short(ap[2*SLM_S]); a10.s3 = as_short(ap[3*SLM_S]);
                a10.s4 = as_short(ap[4*SLM_S]); a10.s5 = as_short(ap[5*SLM_S]);
                a10.s6 = as_short(ap[6*SLM_S]); a10.s7 = as_short(ap[7*SLM_S]);
                ap += 8*SLM_S;
                a20.s0 = as_short(ap[0*SLM_S]); a20.s1 = as_short(ap[1*SLM_S]);
                a20.s2 = as_short(ap[2*SLM_S]); a20.s3 = as_short(ap[3*SLM_S]);
                a20.s4 = as_short(ap[4*SLM_S]); a20.s5 = as_short(ap[5*SLM_S]);
                a20.s6 = as_short(ap[6*SLM_S]); a20.s7 = as_short(ap[7*SLM_S]);
                ap += 8*SLM_S;
                a30.s0 = as_short(ap[0*SLM_S]); a30.s1 = as_short(ap[1*SLM_S]);
                a30.s2 = as_short(ap[2*SLM_S]); a30.s3 = as_short(ap[3*SLM_S]);
                a30.s4 = as_short(ap[4*SLM_S]); a30.s5 = as_short(ap[5*SLM_S]);
                a30.s6 = as_short(ap[6*SLM_S]); a30.s7 = as_short(ap[7*SLM_S]);
            }

            // DPAS k-step 0
            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a00, b0, acc0);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a10, b0, acc1);
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a20, b0, acc2);
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a30, b0, acc3);

            // Load B k-step 1 (k+16..k+31)
            int8 b1;
            {
                __global const half* bp = b_base + 16*N;
                b1.s0 = as_int((half2)(bp[0], bp[N]));
                b1.s1 = as_int((half2)(bp[2*N], bp[3*N]));
                b1.s2 = as_int((half2)(bp[4*N], bp[5*N]));
                b1.s3 = as_int((half2)(bp[6*N], bp[7*N]));
                b1.s4 = as_int((half2)(bp[8*N], bp[9*N]));
                b1.s5 = as_int((half2)(bp[10*N], bp[11*N]));
                b1.s6 = as_int((half2)(bp[12*N], bp[13*N]));
                b1.s7 = as_int((half2)(bp[14*N], bp[15*N]));
            }

            // Load A from SLM k-step 1 (cols 16..31)
            short8 a01, a11, a21, a31;
            {
                __local const half* ap = slm_base + 16 + sg_lid;
                a01.s0 = as_short(ap[0*SLM_S]); a01.s1 = as_short(ap[1*SLM_S]);
                a01.s2 = as_short(ap[2*SLM_S]); a01.s3 = as_short(ap[3*SLM_S]);
                a01.s4 = as_short(ap[4*SLM_S]); a01.s5 = as_short(ap[5*SLM_S]);
                a01.s6 = as_short(ap[6*SLM_S]); a01.s7 = as_short(ap[7*SLM_S]);
                ap += 8*SLM_S;
                a11.s0 = as_short(ap[0*SLM_S]); a11.s1 = as_short(ap[1*SLM_S]);
                a11.s2 = as_short(ap[2*SLM_S]); a11.s3 = as_short(ap[3*SLM_S]);
                a11.s4 = as_short(ap[4*SLM_S]); a11.s5 = as_short(ap[5*SLM_S]);
                a11.s6 = as_short(ap[6*SLM_S]); a11.s7 = as_short(ap[7*SLM_S]);
                ap += 8*SLM_S;
                a21.s0 = as_short(ap[0*SLM_S]); a21.s1 = as_short(ap[1*SLM_S]);
                a21.s2 = as_short(ap[2*SLM_S]); a21.s3 = as_short(ap[3*SLM_S]);
                a21.s4 = as_short(ap[4*SLM_S]); a21.s5 = as_short(ap[5*SLM_S]);
                a21.s6 = as_short(ap[6*SLM_S]); a21.s7 = as_short(ap[7*SLM_S]);
                ap += 8*SLM_S;
                a31.s0 = as_short(ap[0*SLM_S]); a31.s1 = as_short(ap[1*SLM_S]);
                a31.s2 = as_short(ap[2*SLM_S]); a31.s3 = as_short(ap[3*SLM_S]);
                a31.s4 = as_short(ap[4*SLM_S]); a31.s5 = as_short(ap[5*SLM_S]);
                a31.s6 = as_short(ap[6*SLM_S]); a31.s7 = as_short(ap[7*SLM_S]);
            }

            // DPAS k-step 1
            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a01, b1, acc0);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a11, b1, acc1);
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a21, b1, acc2);
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a31, b1, acc3);
        }

        // ===== Phase 1: Compute from buffer 1 (k+32..k+63) =====
        {
            __local const half* slm_base = slm_A + 32 * SLM_S;
            __global const half* b_base = B + (k + 32) * N + sg_col;

            // Load B k-step 0 (k+32..k+47)
            int8 b0;
            {
                __global const half* bp = b_base;
                b0.s0 = as_int((half2)(bp[0], bp[N]));
                b0.s1 = as_int((half2)(bp[2*N], bp[3*N]));
                b0.s2 = as_int((half2)(bp[4*N], bp[5*N]));
                b0.s3 = as_int((half2)(bp[6*N], bp[7*N]));
                b0.s4 = as_int((half2)(bp[8*N], bp[9*N]));
                b0.s5 = as_int((half2)(bp[10*N], bp[11*N]));
                b0.s6 = as_int((half2)(bp[12*N], bp[13*N]));
                b0.s7 = as_int((half2)(bp[14*N], bp[15*N]));
            }

            // Load A from SLM k-step 0 (cols 0..15)
            short8 a00, a10, a20, a30;
            {
                __local const half* ap = slm_base + sg_lid;
                a00.s0 = as_short(ap[0*SLM_S]); a00.s1 = as_short(ap[1*SLM_S]);
                a00.s2 = as_short(ap[2*SLM_S]); a00.s3 = as_short(ap[3*SLM_S]);
                a00.s4 = as_short(ap[4*SLM_S]); a00.s5 = as_short(ap[5*SLM_S]);
                a00.s6 = as_short(ap[6*SLM_S]); a00.s7 = as_short(ap[7*SLM_S]);
                ap += 8*SLM_S;
                a10.s0 = as_short(ap[0*SLM_S]); a10.s1 = as_short(ap[1*SLM_S]);
                a10.s2 = as_short(ap[2*SLM_S]); a10.s3 = as_short(ap[3*SLM_S]);
                a10.s4 = as_short(ap[4*SLM_S]); a10.s5 = as_short(ap[5*SLM_S]);
                a10.s6 = as_short(ap[6*SLM_S]); a10.s7 = as_short(ap[7*SLM_S]);
                ap += 8*SLM_S;
                a20.s0 = as_short(ap[0*SLM_S]); a20.s1 = as_short(ap[1*SLM_S]);
                a20.s2 = as_short(ap[2*SLM_S]); a20.s3 = as_short(ap[3*SLM_S]);
                a20.s4 = as_short(ap[4*SLM_S]); a20.s5 = as_short(ap[5*SLM_S]);
                a20.s6 = as_short(ap[6*SLM_S]); a20.s7 = as_short(ap[7*SLM_S]);
                ap += 8*SLM_S;
                a30.s0 = as_short(ap[0*SLM_S]); a30.s1 = as_short(ap[1*SLM_S]);
                a30.s2 = as_short(ap[2*SLM_S]); a30.s3 = as_short(ap[3*SLM_S]);
                a30.s4 = as_short(ap[4*SLM_S]); a30.s5 = as_short(ap[5*SLM_S]);
                a30.s6 = as_short(ap[6*SLM_S]); a30.s7 = as_short(ap[7*SLM_S]);
            }

            // DPAS k-step 0
            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a00, b0, acc0);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a10, b0, acc1);
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a20, b0, acc2);
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a30, b0, acc3);

            // Load B k-step 1 (k+48..k+63)
            int8 b1;
            {
                __global const half* bp = b_base + 16*N;
                b1.s0 = as_int((half2)(bp[0], bp[N]));
                b1.s1 = as_int((half2)(bp[2*N], bp[3*N]));
                b1.s2 = as_int((half2)(bp[4*N], bp[5*N]));
                b1.s3 = as_int((half2)(bp[6*N], bp[7*N]));
                b1.s4 = as_int((half2)(bp[8*N], bp[9*N]));
                b1.s5 = as_int((half2)(bp[10*N], bp[11*N]));
                b1.s6 = as_int((half2)(bp[12*N], bp[13*N]));
                b1.s7 = as_int((half2)(bp[14*N], bp[15*N]));
            }

            // Load A from SLM k-step 1 (cols 16..31)
            short8 a01, a11, a21, a31;
            {
                __local const half* ap = slm_base + 16 + sg_lid;
                a01.s0 = as_short(ap[0*SLM_S]); a01.s1 = as_short(ap[1*SLM_S]);
                a01.s2 = as_short(ap[2*SLM_S]); a01.s3 = as_short(ap[3*SLM_S]);
                a01.s4 = as_short(ap[4*SLM_S]); a01.s5 = as_short(ap[5*SLM_S]);
                a01.s6 = as_short(ap[6*SLM_S]); a01.s7 = as_short(ap[7*SLM_S]);
                ap += 8*SLM_S;
                a11.s0 = as_short(ap[0*SLM_S]); a11.s1 = as_short(ap[1*SLM_S]);
                a11.s2 = as_short(ap[2*SLM_S]); a11.s3 = as_short(ap[3*SLM_S]);
                a11.s4 = as_short(ap[4*SLM_S]); a11.s5 = as_short(ap[5*SLM_S]);
                a11.s6 = as_short(ap[6*SLM_S]); a11.s7 = as_short(ap[7*SLM_S]);
                ap += 8*SLM_S;
                a21.s0 = as_short(ap[0*SLM_S]); a21.s1 = as_short(ap[1*SLM_S]);
                a21.s2 = as_short(ap[2*SLM_S]); a21.s3 = as_short(ap[3*SLM_S]);
                a21.s4 = as_short(ap[4*SLM_S]); a21.s5 = as_short(ap[5*SLM_S]);
                a21.s6 = as_short(ap[6*SLM_S]); a21.s7 = as_short(ap[7*SLM_S]);
                ap += 8*SLM_S;
                a31.s0 = as_short(ap[0*SLM_S]); a31.s1 = as_short(ap[1*SLM_S]);
                a31.s2 = as_short(ap[2*SLM_S]); a31.s3 = as_short(ap[3*SLM_S]);
                a31.s4 = as_short(ap[4*SLM_S]); a31.s5 = as_short(ap[5*SLM_S]);
                a31.s6 = as_short(ap[6*SLM_S]); a31.s7 = as_short(ap[7*SLM_S]);
            }

            // Start loading next A tiles into both buffers while DPAS executes
            // Load A(k+64) into buffer 0, A(k+96) into buffer 1
            if (k < K_MAIN) {
                __global const half* a_src0 = A + a_row_offset + (k + 64) + a_col_base;
                __local half* a_dst0 = slm_A + a_row * SLM_S + a_col_base;
                half8 v0 = vload8(0, a_src0);
                half8 v1 = vload8(1, a_src0);
                vstore8(v0, 0, a_dst0);
                vstore8(v1, 0, a_dst0 + 8);

                __global const half* a_src1 = A + a_row_offset + (k + 96) + a_col_base;
                __local half* a_dst1 = slm_A + 32 * SLM_S + a_row * SLM_S + a_col_base;
                half8 u0 = vload8(0, a_src1);
                half8 u1 = vload8(1, a_src1);
                vstore8(u0, 0, a_dst1);
                vstore8(u1, 0, a_dst1 + 8);
            }

            // DPAS k-step 1
            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a01, b1, acc0);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a11, b1, acc1);
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a21, b1, acc2);
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a31, b1, acc3);
        }
    }

    // Write C: each WI writes 32 rows to its column
    __global half* c_ptr = C + base_row * N + sg_col;
    c_ptr[0*N] = convert_half(acc0.s0); c_ptr[1*N] = convert_half(acc0.s1);
    c_ptr[2*N] = convert_half(acc0.s2); c_ptr[3*N] = convert_half(acc0.s3);
    c_ptr[4*N] = convert_half(acc0.s4); c_ptr[5*N] = convert_half(acc0.s5);
    c_ptr[6*N] = convert_half(acc0.s6); c_ptr[7*N] = convert_half(acc0.s7);
    c_ptr[8*N] = convert_half(acc1.s0); c_ptr[9*N] = convert_half(acc1.s1);
    c_ptr[10*N] = convert_half(acc1.s2); c_ptr[11*N] = convert_half(acc1.s3);
    c_ptr[12*N] = convert_half(acc1.s4); c_ptr[13*N] = convert_half(acc1.s5);
    c_ptr[14*N] = convert_half(acc1.s6); c_ptr[15*N] = convert_half(acc1.s7);
    c_ptr[16*N] = convert_half(acc2.s0); c_ptr[17*N] = convert_half(acc2.s1);
    c_ptr[18*N] = convert_half(acc2.s2); c_ptr[19*N] = convert_half(acc2.s3);
    c_ptr[20*N] = convert_half(acc2.s4); c_ptr[21*N] = convert_half(acc2.s5);
    c_ptr[22*N] = convert_half(acc2.s6); c_ptr[23*N] = convert_half(acc2.s7);
    c_ptr[24*N] = convert_half(acc3.s0); c_ptr[25*N] = convert_half(acc3.s1);
    c_ptr[26*N] = convert_half(acc3.s2); c_ptr[27*N] = convert_half(acc3.s3);
    c_ptr[28*N] = convert_half(acc3.s4); c_ptr[29*N] = convert_half(acc3.s5);
    c_ptr[30*N] = convert_half(acc3.s6); c_ptr[31*N] = convert_half(acc3.s7);
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

self = <task.TestMatmulOCL object at 0x7188debf23f0>
kernel = <function initialize_matmul_kernel.<locals>.run at 0x718878b391c0>
ocl_queue = <pyopencl._cl.CommandQueue object at 0x718886d3a020>, _run = 0

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
E        +  where False = <function allclose at 0x7188df55ca70>(array([[-1.24375000e+01, -8.52500000e+01,  4.58750000e+01, ...,\n        -6.70625000e+01, -6.45000000e+01,  3.78125000e+01],\n       [-8.29375000e+01,  2.83281250e+01,  4.30859375e+00, ...,\n         3.71875000e+01,  4.88750000e+01,  5.51562500e+01],\n       [ 3.10937500e+01, -5.17812500e+01, -9.30468750e+00, ...,\n         8.12500000e+00,  6.12187500e+01,  4.71093750e+00],\n       ...,\n       [-6.55000000e+01, -2.89687500e+01,  7.55000000e+01, ...,\n         1.22125000e+02, -4.15625000e+01,  1.07109375e+01],\n       [ 4.50937500e+01,  7.40356445e-02,  2.44687500e+01, ...,\n        -3.58125000e+01,  4.27812500e+01,  5.25937500e+01],\n       [ 5.09375000e+01, -6.09765625e+00,  2.36875000e+01, ...,\n        -2.14062500e+01, -3.60312500e+01,  4.90000000e+01]],\n      shape=(2048, 2048), dtype=float32), array([[-12.434087 , -85.22102  ,  45.86866  , ..., -67.074715 ,\n        -64.52674  ,  37.798523 ],\n       [-82.95244  ,  28.332115 ,   4.3084497, ...,  37.17192  ,\n         48.87541  ,  55.1519   ],\n       [ 31.096529 , -51.77693  ,  -9.3054905, ...,   8.124319 ,\n         61.21928  ,   4.7092314],\n       ...,\n       [-65.29967  , -27.73106  ,  74.195465 , ..., 122.09403  ,\n        -41.569603 ,  10.711429 ],\n       [ 44.6838   ,   2.3142765,  22.61605  , ..., -35.807106 ,\n         42.793472 ,  52.60636  ],\n       [ 50.399834 ,  -3.015791 ,  21.545517 , ..., -21.399685 ,\n        -36.035267 ,  49.01544  ]], shape=(2048, 2048), dtype=float32), rtol=0.02, atol=0.02)
E        +    where <function allclose at 0x7188df55ca70> = np.allclose

task.py:244: AssertionError
________________ TestMatmulOCL.test_correctness_wrt_pytorch[1] _________________

self = <task.TestMatmulOCL object at 0x718886cdf5c0>
kernel = <function initialize_matmul_kernel.<locals>.run at 0x718878b391c0>
ocl_queue = <pyopencl._cl.CommandQueue object at 0x718886d3a020>, _run = 1

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
E        +  where False = <function allclose at 0x7188df55ca70>(array([[  63.21875  ,   13.6640625,   62.71875  , ...,   26.953125 ,\n        -100.125    ,  -76.125    ],\n       [  30.34375  ,   -9.578125 ,  -15.8515625, ...,  -86.6875   ,\n           6.3671875,    9.5703125],\n       [  -9.828125 ,   26.84375  ,  -39.875    , ...,   94.3125   ,\n         -40.4375   ,   13.3515625],\n       ...,\n       [ -52.75     ,   -8.109375 ,  -19.46875  , ...,   -4.0625   ,\n         -29.109375 ,   -2.7675781],\n       [ -39.71875  ,   13.0234375,   51.9375   , ...,    3.5292969,\n          14.2578125,   55.59375  ],\n       [ -38.78125  ,   73.375    ,   -8.34375  , ...,   11.921875 ,\n         -64.       ,   63.0625   ]], shape=(2048, 2048), dtype=float32), array([[  63.220627 ,   13.663691 ,   62.708282 , ...,   26.950535 ,\n        -100.14888  ,  -76.10468  ],\n       [  30.338015 ,   -9.576593 ,  -15.848044 , ...,  -86.66203  ,\n           6.3691177,    9.569207 ],\n       [  -9.825886 ,   26.83852  ,  -39.88768  , ...,   94.32298  ,\n         -40.437588 ,   13.349518 ],\n       ...,\n       [ -50.946926 ,  -10.7210655,  -18.652342 , ...,   -4.0612535,\n         -29.112085 ,   -2.7683525],\n       [ -41.46417  ,   -5.034666 ,   35.500336 , ...,    3.5289268,\n          14.26104  ,   55.58531  ],\n       [ -33.896618 ,   51.45737  ,   13.108513 , ...,   11.92079  ,\n         -64.022385 ,   63.048595 ]], shape=(2048, 2048), dtype=float32), rtol=0.02, atol=0.02)
E        +    where <function allclose at 0x7188df55ca70> = np.allclose

task.py:244: AssertionError
_______________ TestMatmulOCL.test_correctness_wrt_reference[0] ________________

self = <task.TestMatmulOCL object at 0x718886d27950>
kernel = <function initialize_matmul_kernel.<locals>.run at 0x718878b391c0>
ocl_queue = <pyopencl._cl.CommandQueue object at 0x718886d3a020>, _run = 0

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
E        +  where False = <function allclose at 0x7188df55ca70>(array([[-5.71250000e+01, -2.80156250e+01,  6.55625000e+01, ...,\n        -4.28437500e+01,  1.19296875e+01, -2.24023438e+00],\n       [ 9.26875000e+01,  1.83437500e+01, -1.43437500e+01, ...,\n        -3.58750000e+01, -5.70312500e+01, -2.84375000e+00],\n       [-2.55859375e+00, -3.00156250e+01,  4.99375000e+01, ...,\n        -4.53437500e+01, -1.12937500e+02,  1.88750000e+01],\n       ...,\n       [-5.01250000e+01, -1.52625000e+02, -5.41250000e+01, ...,\n        -1.22009277e-01,  6.64453125e+00, -5.31562500e+01],\n       [ 3.41250000e+01, -2.00468750e+01,  4.37500000e+01, ...,\n         1.83906250e+01, -2.05078125e+00, -1.60000000e+02],\n       [-4.14062500e+01,  9.46250000e+01,  2.03437500e+01, ...,\n         4.54375000e+01,  5.44375000e+01,  1.07875000e+02]],\n      shape=(2048, 2048), dtype=float32), array([[-5.7125000e+01, -2.8015625e+01,  6.5562500e+01, ...,\n        -5.0218750e+01,  2.7000000e+01,  2.6109375e+01],\n       [ 9.2687500e+01,  1.8343750e+01, -1.4343750e+01, ...,\n        -4.0843750e+01, -2.5828125e+01, -3.1914062e+00],\n       [-2.5585938e+00, -3.0015625e+01,  4.9937500e+01, ...,\n        -5.7156250e+01, -9.2250000e+01,  1.2921875e+01],\n       ...,\n       [-4.6031250e+01, -1.4262500e+02, -6.1500000e+01, ...,\n        -1.2194824e-01,  6.6445312e+00, -5.3156250e+01],\n       [ 3.1859375e+01, -2.3484375e+01,  4.2750000e+01, ...,\n         1.8390625e+01, -2.0507812e+00, -1.6000000e+02],\n       [-3.8968750e+01,  9.1750000e+01,  2.3953125e+01, ...,\n         4.5437500e+01,  5.4437500e+01,  1.0787500e+02]],\n      shape=(2048, 2048), dtype=float32), rtol=0.02, atol=0.02)
E        +    where <function allclose at 0x7188df55ca70> = np.allclose

task.py:262: AssertionError
_______________ TestMatmulOCL.test_correctness_wrt_reference[1] ________________

self = <task.TestMatmulOCL object at 0x718886d27a10>
kernel = <function initialize_matmul_kernel.<locals>.run at 0x718878b391c0>
ocl_queue = <pyopencl._cl.CommandQueue object at 0x718886d3a020>, _run = 1

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
E        +  where False = <function allclose at 0x7188df55ca70>(array([[  46.4375   ,  -21.75     ,  -36.71875  , ...,  -39.96875  ,\n           3.125    , -147.25     ],\n       [ -13.859375 ,  -17.453125 ,   17.375    , ...,   57.90625  ,\n          25.859375 ,  -68.0625   ],\n       [  34.1875   ,   -9.421875 ,  -49.       , ...,  -46.71875  ,\n         -64.5625   ,   37.6875   ],\n       ...,\n       [  22.984375 ,  -59.15625  ,  -52.03125  , ...,   18.328125 ,\n         -17.234375 ,  -30.90625  ],\n       [  20.984375 ,   99.1875   ,  -13.3984375, ...,   35.75     ,\n         -45.0625   ,   21.78125  ],\n       [ -21.390625 ,  -58.875    ,   72.875    , ...,  -48.09375  ,\n         -29.515625 ,   11.1640625]], shape=(2048, 2048), dtype=float32), array([[  46.4375   ,  -21.75     ,  -36.71875  , ...,  -39.96875  ,\n           3.125    , -147.25     ],\n       [ -13.859375 ,  -17.453125 ,   17.375    , ...,   57.90625  ,\n          25.859375 ,  -68.0625   ],\n       [  34.1875   ,   -9.421875 ,  -49.       , ...,  -46.71875  ,\n         -64.5625   ,   37.6875   ],\n       ...,\n       [   3.3261719,  -60.03125  ,  -64.       , ...,   18.328125 ,\n         -17.234375 ,  -30.90625  ],\n       [  -0.9145508,   97.25     ,  -18.109375 , ...,   35.75     ,\n         -45.0625   ,   21.78125  ],\n       [ -36.5      ,  -63.8125   ,   68.125    , ...,  -48.09375  ,\n         -29.515625 ,   11.1640625]], shape=(2048, 2048), dtype=float32), rtol=0.02, atol=0.02)
E        +    where <function allclose at 0x7188df55ca70> = np.allclose

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
================== 4 failed, 1 deselected, 1 warning in 0.93s ==================

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

- **Register Blocking**: Each work-item computes a THREAD_M×THREAD_N output block in private register arrays. Use `#pragma unroll` on inner loops. Combine with SLM tiling for multi-level memory hierarchy optimization.
- **Sub-Group Collectives**: Use `reduce_over_group(sg, val, op)` for hardware-accelerated SIMD reductions. Use `group_broadcast` and `shift_group_*` for efficient data sharing.