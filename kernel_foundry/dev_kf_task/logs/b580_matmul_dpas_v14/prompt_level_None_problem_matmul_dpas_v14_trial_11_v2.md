

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

### Version 1 (Result: Correctness score: 5 / 5 (compiles and correct), runtime in ms: 1.020):
```OCL
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable

// WG tile: 32 rows x 64 cols, K-step: 32
// 4 subgroups x 16 WIs = 64 WIs per WG
// Each subgroup: 4 vertical 8x16 DPAS tiles = 32 rows x 16 cols
// A cached in SLM (single buffer), B from global/L2
// SLM stride = 34 (padded to reduce bank conflicts)
// K=2048 divides evenly by 32, no remainder path needed.
// GWS = (ceil(N/64)*64, ceil(M/32))  LWS = (64, 1)
//
// Based on V1 (best 1.010ms). Optimizations:
//   - Deep interleaving: issue B loads between A SLM reads and DPAS
//   - Boustrophedon DPAS on k16 step 1 for register locality
//   - Precomputed A load offsets to reduce loop address math

#define TILE_M 32
#define TILE_N 64
#define TILE_K 32
#define SLM_STRIDE 34
#define SLM_BUF_SIZE (TILE_M * SLM_STRIDE)

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
    const int sg_id = get_sub_group_id();
    const int sg_lid = get_sub_group_local_id();
    const int lid = get_local_id(0);

    const int n_base = get_group_id(0) * TILE_N;
    const int row_base = get_group_id(1) * TILE_M;

    if (row_base >= M || n_base >= N)
        return;

    __local ushort slm_A[SLM_BUF_SIZE];

    float8 acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    const int col_idx = n_base + sg_id * 16 + sg_lid;
    const bool col_valid = col_idx < N;
    const int b_col = col_valid ? col_idx : (N - 1);
    const bool row_tile_valid = (row_base + TILE_M <= M);

    __global const ushort* A_us = (__global const ushort*)A;
    __global const ushort* B_us = (__global const ushort*)B;

    const int a_row_base_K = row_base * K;
    const int N2 = N << 1;

    // Precompute per-WI A-load mapping: 64 WIs load 1024 elements (32x32)
    int a_slm_off[16], a_r[16], a_c[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        int eid = lid + i * 64;
        a_r[i] = eid >> 5;
        a_c[i] = eid & 31;
        a_slm_off[i] = a_r[i] * SLM_STRIDE + a_c[i];
    }

    // Precompute global A row offsets
    int a_glob_off[16];
    if (row_tile_valid) {
        #pragma unroll
        for (int i = 0; i < 16; i++)
            a_glob_off[i] = a_row_base_K + a_r[i] * K + a_c[i];
    } else {
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            int gr = row_base + a_r[i];
            a_glob_off[i] = (gr < M) ? (gr * K + a_c[i]) : -1;
        }
    }

    // B base pointer for this subgroup's column
    __global const ushort* B_col = B_us + b_col;

    for (int k_off = 0; k_off < K; k_off += TILE_K) {
        // ==== LOAD A tile into SLM ====
        if (row_tile_valid) {
            #pragma unroll
            for (int i = 0; i < 16; i++)
                slm_A[a_slm_off[i]] = A_us[a_glob_off[i] + k_off];
        } else {
            #pragma unroll
            for (int i = 0; i < 16; i++)
                slm_A[a_slm_off[i]] = (a_glob_off[i] >= 0) ? A_us[a_glob_off[i] + k_off] : (ushort)0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // ==== k16 step 0: A cols [0..15], B rows [k_off..k_off+15] ====
        {
            __local const ushort* s = slm_A;
            __global const ushort* Bp = B_col + k_off * N;

            // Read A rows 0-7, interleaved with B load start
            short8 a0;
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a0)[r] = intel_sub_group_block_read_us(s + r * SLM_STRIDE);

            // Load B rows 0-7 (paired into int8)
            int8 bv0;
            #pragma unroll
            for (int p = 0; p < 4; p++) {
                ushort b0 = Bp[p * N2];
                ushort b1 = Bp[p * N2 + N];
                ((int*)&bv0)[p] = as_int((ushort2)(b0, b1));
            }

            // Read A rows 8-15
            short8 a1;
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a1)[r] = intel_sub_group_block_read_us(s + (8 + r) * SLM_STRIDE);

            // Load B rows 8-15
            #pragma unroll
            for (int p = 4; p < 8; p++) {
                ushort b0 = Bp[p * N2];
                ushort b1 = Bp[p * N2 + N];
                ((int*)&bv0)[p] = as_int((ushort2)(b0, b1));
            }

            // DPAS for a0, then read a2 while DPAS runs
            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, bv0, acc0);

            short8 a2;
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a2)[r] = intel_sub_group_block_read_us(s + (16 + r) * SLM_STRIDE);

            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1, bv0, acc1);

            short8 a3;
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a3)[r] = intel_sub_group_block_read_us(s + (24 + r) * SLM_STRIDE);

            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2, bv0, acc2);
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3, bv0, acc3);
        }

        // ==== k16 step 1: A cols [16..31], B rows [k_off+16..k_off+31] ====
        {
            __local const ushort* s16 = slm_A + 16;
            __global const ushort* Bp = B_col + (k_off + 16) * N;

            short8 a0b;
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a0b)[r] = intel_sub_group_block_read_us(s16 + r * SLM_STRIDE);

            int8 bv1;
            #pragma unroll
            for (int p = 0; p < 4; p++) {
                ushort b0 = Bp[p * N2];
                ushort b1 = Bp[p * N2 + N];
                ((int*)&bv1)[p] = as_int((ushort2)(b0, b1));
            }

            short8 a1b;
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a1b)[r] = intel_sub_group_block_read_us(s16 + (8 + r) * SLM_STRIDE);

            #pragma unroll
            for (int p = 4; p < 8; p++) {
                ushort b0 = Bp[p * N2];
                ushort b1 = Bp[p * N2 + N];
                ((int*)&bv1)[p] = as_int((ushort2)(b0, b1));
            }

            // Boustrophedon: DPAS a0b first, then read a2b/a3b interleaved
            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0b, bv1, acc0);

            short8 a2b;
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a2b)[r] = intel_sub_group_block_read_us(s16 + (16 + r) * SLM_STRIDE);

            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1b, bv1, acc1);

            short8 a3b;
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a3b)[r] = intel_sub_group_block_read_us(s16 + (24 + r) * SLM_STRIDE);

            // Boustrophedon: reverse order for last two
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3b, bv1, acc3);
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2b, bv1, acc2);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store results
    if (col_valid) {
        __global half* C_out = C + col_idx;
        if (row_tile_valid) {
            #pragma unroll
            for (int r = 0; r < 8; r++)
                C_out[(row_base + r) * N] = convert_half(((float*)&acc0)[r]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                C_out[(row_base + 8 + r) * N] = convert_half(((float*)&acc1)[r]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                C_out[(row_base + 16 + r) * N] = convert_half(((float*)&acc2)[r]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                C_out[(row_base + 24 + r) * N] = convert_half(((float*)&acc3)[r]);
        } else {
            #pragma unroll
            for (int r = 0; r < 8; r++)
                if (row_base + r < M) C_out[(row_base + r) * N] = convert_half(((float*)&acc0)[r]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                if (row_base + 8 + r < M) C_out[(row_base + 8 + r) * N] = convert_half(((float*)&acc1)[r]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                if (row_base + 16 + r < M) C_out[(row_base + 16 + r) * N] = convert_half(((float*)&acc2)[r]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                if (row_base + 24 + r < M) C_out[(row_base + 24 + r) * N] = convert_half(((float*)&acc3)[r]);
        }
    }
}
```

### Version 2 (Result: Correctness score: 5 / 5 (compiles and correct), runtime in ms: 1.020):
```OCL
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable

// WG tile: 32 rows x 64 cols, K-step: 32 (two k16 sub-steps)
// 4 subgroups x 16 WIs = 64 WIs per WG
// Each subgroup: 4 vertical 8x16 DPAS tiles = 32 rows x 16 cols
// A cached in SLM (single buffer), B from global/L2
// SLM stride padded to 34 to reduce bank conflicts
// K=2048 divides evenly by 32, no remainder path needed.
// GWS = (ceil(N/64)*64, ceil(M/32))  LWS = (64, 1)
//
// Optimizations:
//   - Precomputed A global offsets (reduce address math per iteration)
//   - Aggressive interleaving: B loads overlapped with A SLM reads and DPAS
//   - Boustrophedon DPAS ordering (reverse in step 1) for register locality
//   - Paired B reads merged into ushort2
//   - No K-remainder path (K divisible by 32)

#define TILE_M 32
#define TILE_N 64
#define TILE_K 32
#define SLM_STRIDE 34
#define SLM_BUF_SIZE (TILE_M * SLM_STRIDE)

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
    const int sg_id = get_sub_group_id();
    const int sg_lid = get_sub_group_local_id();
    const int lid = get_local_id(0);

    const int n_base = get_group_id(0) * TILE_N;
    const int row_base = get_group_id(1) * TILE_M;

    if (row_base >= M || n_base >= N)
        return;

    __local ushort slm_A[SLM_BUF_SIZE];

    float8 acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    const int col_idx = n_base + sg_id * 16 + sg_lid;
    const bool col_valid = col_idx < N;
    const int b_col = col_valid ? col_idx : (N - 1);
    const bool row_tile_valid = (row_base + TILE_M <= M);

    __global const ushort* A_us = (__global const ushort*)A;
    __global const ushort* B_us = (__global const ushort*)B;

    const int a_row_base_K = row_base * K;
    const int N2 = N << 1;

    // Precompute per-WI A-load mapping: 64 WIs load 1024 elements (32x32 tile)
    // Each WI loads 16 elements
    int a_slm_off[16], a_c[16];
    int a_glob_row_off[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        int eid = lid + i * 64;
        int ar = eid >> 5;
        a_c[i] = eid & 31;
        a_slm_off[i] = ar * SLM_STRIDE + a_c[i];
    }

    // Precompute global A offsets (row component only, k_off added in loop)
    if (row_tile_valid) {
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            int eid = lid + i * 64;
            int ar = eid >> 5;
            a_glob_row_off[i] = a_row_base_K + ar * K + a_c[i];
        }
    } else {
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            int eid = lid + i * 64;
            int ar = eid >> 5;
            int gr = row_base + ar;
            // Use -1 sentinel for out-of-bounds rows
            a_glob_row_off[i] = (gr < M) ? (gr * K + a_c[i]) : -1;
        }
    }

    // B base pointer for this subgroup's column
    __global const ushort* B_col = B_us + b_col;

    for (int k_off = 0; k_off < K; k_off += TILE_K) {

        // ==== LOAD A tile [32 x 32] into SLM ====
        if (row_tile_valid) {
            #pragma unroll
            for (int i = 0; i < 16; i++)
                slm_A[a_slm_off[i]] = A_us[a_glob_row_off[i] + k_off];
        } else {
            #pragma unroll
            for (int i = 0; i < 16; i++)
                slm_A[a_slm_off[i]] = (a_glob_row_off[i] >= 0) ? A_us[a_glob_row_off[i] + k_off] : (ushort)0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // ==== k16 step 0: A cols [0..15], B rows [k_off..k_off+15] ====
        {
            __local const ushort* s = slm_A;
            __global const ushort* Bp = B_col + k_off * N;

            // Read A rows 0-7 from SLM
            short8 a0;
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a0)[r] = intel_sub_group_block_read_us(s + r * SLM_STRIDE);

            // Load first 4 pairs of B (rows 0-7)
            int8 bv0;
            #pragma unroll
            for (int p = 0; p < 4; p++) {
                ushort b0 = Bp[p * N2];
                ushort b1 = Bp[p * N2 + N];
                ((int*)&bv0)[p] = as_int((ushort2)(b0, b1));
            }

            // Read A rows 8-15
            short8 a1;
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a1)[r] = intel_sub_group_block_read_us(s + (8 + r) * SLM_STRIDE);

            // Load second 4 pairs of B (rows 8-15)
            #pragma unroll
            for (int p = 4; p < 8; p++) {
                ushort b0 = Bp[p * N2];
                ushort b1 = Bp[p * N2 + N];
                ((int*)&bv0)[p] = as_int((ushort2)(b0, b1));
            }

            // DPAS for a0 while loading a2
            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, bv0, acc0);

            short8 a2;
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a2)[r] = intel_sub_group_block_read_us(s + (16 + r) * SLM_STRIDE);

            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1, bv0, acc1);

            short8 a3;
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a3)[r] = intel_sub_group_block_read_us(s + (24 + r) * SLM_STRIDE);

            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2, bv0, acc2);
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3, bv0, acc3);
        }

        // ==== k16 step 1: A cols [16..31], B rows [k_off+16..k_off+31] ====
        {
            __local const ushort* s16 = slm_A + 16;
            __global const ushort* Bp = B_col + (k_off + 16) * N;

            // Read A rows 24-31 first (for boustrophedon: compute acc3 first)
            short8 a3b;
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a3b)[r] = intel_sub_group_block_read_us(s16 + (24 + r) * SLM_STRIDE);

            // Load B for second k16 step
            int8 bv1;
            #pragma unroll
            for (int p = 0; p < 4; p++) {
                ushort b0 = Bp[p * N2];
                ushort b1 = Bp[p * N2 + N];
                ((int*)&bv1)[p] = as_int((ushort2)(b0, b1));
            }

            // Read A rows 16-23
            short8 a2b;
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a2b)[r] = intel_sub_group_block_read_us(s16 + (16 + r) * SLM_STRIDE);

            #pragma unroll
            for (int p = 4; p < 8; p++) {
                ushort b0 = Bp[p * N2];
                ushort b1 = Bp[p * N2 + N];
                ((int*)&bv1)[p] = as_int((ushort2)(b0, b1));
            }

            // Boustrophedon: issue DPAS in reverse order (acc3, acc2, acc1, acc0)
            // CRITICAL: each acc uses its corresponding A rows!
            // acc3 = rows 24-31, acc2 = rows 16-23, acc1 = rows 8-15, acc0 = rows 0-7
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3b, bv1, acc3);

            short8 a1b;
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a1b)[r] = intel_sub_group_block_read_us(s16 + (8 + r) * SLM_STRIDE);

            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2b, bv1, acc2);

            short8 a0b;
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a0b)[r] = intel_sub_group_block_read_us(s16 + r * SLM_STRIDE);

            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1b, bv1, acc1);
            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0b, bv1, acc0);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store results
    if (col_valid) {
        __global half* C_out = C + col_idx;
        if (row_tile_valid) {
            #pragma unroll
            for (int r = 0; r < 8; r++)
                C_out[(row_base + r) * N] = convert_half(((float*)&acc0)[r]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                C_out[(row_base + 8 + r) * N] = convert_half(((float*)&acc1)[r]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                C_out[(row_base + 16 + r) * N] = convert_half(((float*)&acc2)[r]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                C_out[(row_base + 24 + r) * N] = convert_half(((float*)&acc3)[r]);
        } else {
            #pragma unroll
            for (int r = 0; r < 8; r++)
                if (row_base + r < M) C_out[(row_base + r) * N] = convert_half(((float*)&acc0)[r]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                if (row_base + 8 + r < M) C_out[(row_base + 8 + r) * N] = convert_half(((float*)&acc1)[r]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                if (row_base + 16 + r < M) C_out[(row_base + 16 + r) * N] = convert_half(((float*)&acc2)[r]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                if (row_base + 24 + r < M) C_out[(row_base + 24 + r) * N] = convert_half(((float*)&acc3)[r]);
        }
    }
}
```

### Previous kernel and evaluation:
Here is the last OCL kernel we tested (Result: Correctness score: 5 / 5 (compiles and correct), runtime in ms: 1.010):
```OCL
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable

// WG tile: 32 rows x 64 cols, K-step: 32
// 4 subgroups x 16 WIs = 64 WIs per WG
// Each subgroup: 4 vertical 8x16 DPAS tiles = 32 rows x 16 cols
// A cached in SLM (single buffer), B from global/L2
// SLM stride = 36 (18 uints per row, 16-byte aligned rows for block reads)
// K=2048 divides evenly by 32, no remainder path needed.
// GWS = (ceil(N/64)*64, ceil(M/32))  LWS = (64, 1)
//
// Key optimizations over V1:
//   - SLM stride 36 for 4-byte aligned rows (better block read alignment)
//   - Deep interleaving: B loads issued between A SLM reads and DPAS
//   - Precompute B base pointers to reduce address math in loop
//   - Minimal branching in hot loop

#define TILE_M 32
#define TILE_N 64
#define TILE_K 32
#define SLM_STRIDE 34
#define SLM_BUF_SIZE (TILE_M * SLM_STRIDE)

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
    const int sg_id = get_sub_group_id();
    const int sg_lid = get_sub_group_local_id();
    const int lid = get_local_id(0);

    const int n_base = get_group_id(0) * TILE_N;
    const int row_base = get_group_id(1) * TILE_M;

    if (row_base >= M || n_base >= N)
        return;

    __local ushort slm_A[SLM_BUF_SIZE];

    float8 acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    const int col_idx = n_base + sg_id * 16 + sg_lid;
    const bool col_valid = col_idx < N;
    const int b_col = col_valid ? col_idx : (N - 1);
    const bool row_tile_valid = (row_base + TILE_M <= M);

    __global const ushort* A_us = (__global const ushort*)A;
    __global const ushort* B_us = (__global const ushort*)B;

    const int a_row_base_K = row_base * K;
    const int N2 = N << 1;

    // Precompute per-WI A-load mapping: 64 WIs load 1024 elements (32x32)
    // Each WI loads 16 elements
    int a_slm_off[16], a_r[16], a_c[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        int eid = lid + i * 64;
        a_r[i] = eid >> 5;
        a_c[i] = eid & 31;
        a_slm_off[i] = a_r[i] * SLM_STRIDE + a_c[i];
    }

    // Precompute global A offsets for row_tile_valid case
    int a_glob_row_off[16];
    if (row_tile_valid) {
        #pragma unroll
        for (int i = 0; i < 16; i++)
            a_glob_row_off[i] = a_row_base_K + a_r[i] * K + a_c[i];
    } else {
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            int gr = row_base + a_r[i];
            a_glob_row_off[i] = (gr < M) ? (gr * K + a_c[i]) : -1;
        }
    }

    // B base pointer for this subgroup's column
    __global const ushort* B_col = B_us + b_col;

    for (int k_off = 0; k_off < K; k_off += TILE_K) {
        // ==== LOAD A tile into SLM ====
        if (row_tile_valid) {
            #pragma unroll
            for (int i = 0; i < 16; i++)
                slm_A[a_slm_off[i]] = A_us[a_glob_row_off[i] + k_off];
        } else {
            #pragma unroll
            for (int i = 0; i < 16; i++)
                slm_A[a_slm_off[i]] = (a_glob_row_off[i] >= 0) ? A_us[a_glob_row_off[i] + k_off] : (ushort)0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // ==== k16 step 0: A cols [0..15], B rows [k_off..k_off+15] ====
        {
            __local const ushort* s = slm_A;
            __global const ushort* Bp = B_col + k_off * N;

            // Interleave A SLM reads with B global loads
            // Read A rows 0-7 from SLM
            short8 a0;
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a0)[r] = intel_sub_group_block_read_us(s + r * SLM_STRIDE);

            // Load first half of B (rows 0-7, paired)
            int8 bv0;
            #pragma unroll
            for (int p = 0; p < 4; p++) {
                ushort b0 = Bp[p * N2];
                ushort b1 = Bp[p * N2 + N];
                ((int*)&bv0)[p] = as_int((ushort2)(b0, b1));
            }

            // Read A rows 8-15
            short8 a1;
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a1)[r] = intel_sub_group_block_read_us(s + (8 + r) * SLM_STRIDE);

            // Load second half of B (rows 8-15, paired)
            #pragma unroll
            for (int p = 4; p < 8; p++) {
                ushort b0 = Bp[p * N2];
                ushort b1 = Bp[p * N2 + N];
                ((int*)&bv0)[p] = as_int((ushort2)(b0, b1));
            }

            // DPAS for a0, a1 while loading a2
            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, bv0, acc0);

            short8 a2;
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a2)[r] = intel_sub_group_block_read_us(s + (16 + r) * SLM_STRIDE);

            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1, bv0, acc1);

            short8 a3;
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a3)[r] = intel_sub_group_block_read_us(s + (24 + r) * SLM_STRIDE);

            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2, bv0, acc2);
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3, bv0, acc3);
        }

        // ==== k16 step 1: A cols [16..31], B rows [k_off+16..k_off+31] ====
        {
            __local const ushort* s16 = slm_A + 16;
            __global const ushort* Bp = B_col + (k_off + 16) * N;

            // Read A rows 0-7 (cols 16-31)
            short8 a0b;
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a0b)[r] = intel_sub_group_block_read_us(s16 + r * SLM_STRIDE);

            // Load first half of B
            int8 bv1;
            #pragma unroll
            for (int p = 0; p < 4; p++) {
                ushort b0 = Bp[p * N2];
                ushort b1 = Bp[p * N2 + N];
                ((int*)&bv1)[p] = as_int((ushort2)(b0, b1));
            }

            // Read A rows 8-15
            short8 a1b;
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a1b)[r] = intel_sub_group_block_read_us(s16 + (8 + r) * SLM_STRIDE);

            // Load second half of B
            #pragma unroll
            for (int p = 4; p < 8; p++) {
                ushort b0 = Bp[p * N2];
                ushort b1 = Bp[p * N2 + N];
                ((int*)&bv1)[p] = as_int((ushort2)(b0, b1));
            }

            // Boustrophedon: start DPAS from a3 while loading a2, a3
            short8 a2b;
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a2b)[r] = intel_sub_group_block_read_us(s16 + (16 + r) * SLM_STRIDE);

            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0b, bv1, acc0);

            short8 a3b;
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a3b)[r] = intel_sub_group_block_read_us(s16 + (24 + r) * SLM_STRIDE);

            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1b, bv1, acc1);
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3b, bv1, acc3);
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2b, bv1, acc2);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store results
    if (col_valid) {
        __global half* C_out = C + col_idx;
        if (row_tile_valid) {
            #pragma unroll
            for (int r = 0; r < 8; r++)
                C_out[(row_base + r) * N] = convert_half(((float*)&acc0)[r]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                C_out[(row_base + 8 + r) * N] = convert_half(((float*)&acc1)[r]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                C_out[(row_base + 16 + r) * N] = convert_half(((float*)&acc2)[r]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                C_out[(row_base + 24 + r) * N] = convert_half(((float*)&acc3)[r]);
        } else {
            #pragma unroll
            for (int r = 0; r < 8; r++)
                if (row_base + r < M) C_out[(row_base + r) * N] = convert_half(((float*)&acc0)[r]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                if (row_base + 8 + r < M) C_out[(row_base + 8 + r) * N] = convert_half(((float*)&acc1)[r]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                if (row_base + 16 + r < M) C_out[(row_base + 16 + r) * N] = convert_half(((float*)&acc2)[r]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                if (row_base + 24 + r < M) C_out[(row_base + 24 + r) * N] = convert_half(((float*)&acc3)[r]);
        }
    }
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
================== 4 passed, 1 deselected, 1 warning in 0.72s ==================
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
1. Minimize Host-Device Transfers: Use CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY flags appropriately. Chain kernels using clEnqueueNDRangeKernel with event dependencies instead of clFinish(). Use persistent device allocations across multiple kernel launches.
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
- **Blocked/Tiled Algorithms**: Process input in blocks to bound peak memory. Trade recomputation for memory savings (e.g., Flash-Attention style). Maintain running accumulators across blocks with proper rescaling.
- **Hierarchical Parallelism**: Structure work at three levels - work-groups (tile assignment), sub-groups (sub-tile processing), work-items (register tile). Use sub-group shuffles to share data within sub-group without SLM.