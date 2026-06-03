

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

### Version 1 (Result: Correctness score: 5 / 5 (compiles and correct), runtime in ms: 1.410):
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
    // 4 subgroups, each handles 32 rows x 16 cols of C
    const int sg_id = get_local_id(1);       // 0..3 (which subgroup)
    const int sg_lid = get_sub_group_local_id(); // 0..15 (lane within SG)

    // WG position in output
    const int wg_row = get_group_id(1) * 32;  // M tile start
    const int wg_col = get_group_id(0) * 64;  // N tile start (get_group_id(0) = global_id(0)/16... no, with LWS=(16,4,1) group_id(0) covers N/64)

    // Wait - with LWS=(16,4,1) and GWS=(N/4, M/8, 1):
    // get_group_id(0) = 0..N/64-1 (since GWS_x/LWS_x = (N/4)/16 = N/64)
    // get_group_id(1) = 0..M/32-1 (since GWS_y/LWS_y = (M/8)/4 = M/32)
    // This is correct.

    // Each subgroup handles 16 consecutive columns
    const int sg_col = wg_col + sg_id * 16;

    // SLM for A tile: double-buffered, 32 rows x 32 cols (half)
    // Use stride of 32 for efficient block reads from SLM
    // Total SLM: 2 * 32 * 32 * 2 bytes = 4096 bytes (fits easily in 64KB)
    #define SLM_STRIDE 32
    __local half slm_A[2 * 32 * SLM_STRIDE];

    // Accumulators: 32 rows x 16 cols per subgroup = 4 DPAS results of float8
    float8 acc0 = 0.0f;  // rows 0-7
    float8 acc1 = 0.0f;  // rows 8-15
    float8 acc2 = 0.0f;  // rows 16-23
    float8 acc3 = 0.0f;  // rows 24-31

    // Linear local ID for cooperative loading
    const int lid = sg_id * 16 + sg_lid;  // 0..63

    // A load mapping: 64 WIs load 32x32 = 1024 halfs = 16 per WI
    // lid/2 = row (0..31), (lid%2)*16 = col offset (0 or 16)
    const int a_load_row = lid / 2;
    const int a_load_col_off = (lid & 1) * 16;

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

    // Main K-loop, step by 32
    // K is guaranteed to be divisible by 32
    for (int k = 0; k < K; k += 32) {
        int next_k = k + 32;

        // === COMPUTE from current SLM buffer ===
        int slm_base = buf * 32 * SLM_STRIDE;

        // B pointer for this K-step
        __global const half* b_base_ptr = B + k * N + sg_col + sg_lid;

        // --- K16 step 0 (k=0..15 within tile) ---
        {
            // Load A from SLM using intel_sub_group_block_read_us
            // For rows r*8..(r*8+7), k=0..15:
            // SLM layout: slm_A[slm_base + row*SLM_STRIDE + k_col]
            // With SLM_STRIDE=32 and reading 16 consecutive elements per row,
            // and stride between rows = 32 halfs = 64 bytes
            // intel_sub_group_block_read_us8 reads 8*16 = 128 ushorts contiguously
            // That means it reads 128 consecutive ushorts = 256 bytes
            // With stride=32 between rows (in halfs), rows are at offsets 0, 32, 64, ...
            // block_read_us8 reads at offsets: lane + {0,16,32,48,64,80,96,112}
            // With SLM_STRIDE=32: row 0 k=0..15 at offset 0..15, row 0 k=16..31 at 16..31
            //                     row 1 k=0..15 at offset 32..47
            // So block_read_us8 from offset 0 gives:
            //   vec[0] = slm[lane] = row 0, k=lane (0..15) ✓
            //   vec[1] = slm[16+lane] = row 0, k=16+lane (16..31) ✗ (we want row 1, k=lane)
            // This doesn't work with stride=32 directly.

            // With SLM_STRIDE=16: we'd store A as two 32x16 blocks
            // But that complicates the store pattern.

            // Alternative: just use scalar SLM reads (they're fast from SLM)
            // Each lane reads its k-column for 8 rows

            short8 a0, a1, a2, a3;

            // rows 0-7
            __local const half* slm_ptr = &slm_A[slm_base + sg_lid];
            a0.s0 = as_short(slm_ptr[0*SLM_STRIDE]);
            a0.s1 = as_short(slm_ptr[1*SLM_STRIDE]);
            a0.s2 = as_short(slm_ptr[2*SLM_STRIDE]);
            a0.s3 = as_short(slm_ptr[3*SLM_STRIDE]);
            a0.s4 = as_short(slm_ptr[4*SLM_STRIDE]);
            a0.s5 = as_short(slm_ptr[5*SLM_STRIDE]);
            a0.s6 = as_short(slm_ptr[6*SLM_STRIDE]);
            a0.s7 = as_short(slm_ptr[7*SLM_STRIDE]);

            // rows 8-15
            a1.s0 = as_short(slm_ptr[8*SLM_STRIDE]);
            a1.s1 = as_short(slm_ptr[9*SLM_STRIDE]);
            a1.s2 = as_short(slm_ptr[10*SLM_STRIDE]);
            a1.s3 = as_short(slm_ptr[11*SLM_STRIDE]);
            a1.s4 = as_short(slm_ptr[12*SLM_STRIDE]);
            a1.s5 = as_short(slm_ptr[13*SLM_STRIDE]);
            a1.s6 = as_short(slm_ptr[14*SLM_STRIDE]);
            a1.s7 = as_short(slm_ptr[15*SLM_STRIDE]);

            // rows 16-23
            a2.s0 = as_short(slm_ptr[16*SLM_STRIDE]);
            a2.s1 = as_short(slm_ptr[17*SLM_STRIDE]);
            a2.s2 = as_short(slm_ptr[18*SLM_STRIDE]);
            a2.s3 = as_short(slm_ptr[19*SLM_STRIDE]);
            a2.s4 = as_short(slm_ptr[20*SLM_STRIDE]);
            a2.s5 = as_short(slm_ptr[21*SLM_STRIDE]);
            a2.s6 = as_short(slm_ptr[22*SLM_STRIDE]);
            a2.s7 = as_short(slm_ptr[23*SLM_STRIDE]);

            // rows 24-31
            a3.s0 = as_short(slm_ptr[24*SLM_STRIDE]);
            a3.s1 = as_short(slm_ptr[25*SLM_STRIDE]);
            a3.s2 = as_short(slm_ptr[26*SLM_STRIDE]);
            a3.s3 = as_short(slm_ptr[27*SLM_STRIDE]);
            a3.s4 = as_short(slm_ptr[28*SLM_STRIDE]);
            a3.s5 = as_short(slm_ptr[29*SLM_STRIDE]);
            a3.s6 = as_short(slm_ptr[30*SLM_STRIDE]);
            a3.s7 = as_short(slm_ptr[31*SLM_STRIDE]);

            // Load B[k:k+16, sg_col+sg_lid] and pack into VNNI int8
            // B is row-major: B[row][col] at B[row*N + col]
            // We need pairs: b[i] = pack(B[k+2i][col], B[k+2i+1][col])
            int8 b_reg;
            __global const half* bp = b_base_ptr;

            half h0, h1;
            h0 = bp[0]; h1 = bp[N];
            b_reg.s0 = as_int((short2)(as_short(h0), as_short(h1)));
            h0 = bp[2*N]; h1 = bp[3*N];
            b_reg.s1 = as_int((short2)(as_short(h0), as_short(h1)));
            h0 = bp[4*N]; h1 = bp[5*N];
            b_reg.s2 = as_int((short2)(as_short(h0), as_short(h1)));
            h0 = bp[6*N]; h1 = bp[7*N];
            b_reg.s3 = as_int((short2)(as_short(h0), as_short(h1)));
            h0 = bp[8*N]; h1 = bp[9*N];
            b_reg.s4 = as_int((short2)(as_short(h0), as_short(h1)));
            h0 = bp[10*N]; h1 = bp[11*N];
            b_reg.s5 = as_int((short2)(as_short(h0), as_short(h1)));
            h0 = bp[12*N]; h1 = bp[13*N];
            b_reg.s6 = as_int((short2)(as_short(h0), as_short(h1)));
            h0 = bp[14*N]; h1 = bp[15*N];
            b_reg.s7 = as_int((short2)(as_short(h0), as_short(h1)));

            // DPAS calls
            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b_reg, acc0);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b_reg, acc1);
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b_reg, acc2);
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b_reg, acc3);
        }

        // --- K16 step 1 (k=16..31 within tile) ---
        {
            short8 a0, a1, a2, a3;

            __local const half* slm_ptr = &slm_A[slm_base + 16 + sg_lid];
            a0.s0 = as_short(slm_ptr[0*SLM_STRIDE]);
            a0.s1 = as_short(slm_ptr[1*SLM_STRIDE]);
            a0.s2 = as_short(slm_ptr[2*SLM_STRIDE]);
            a0.s3 = as_short(slm_ptr[3*SLM_STRIDE]);
            a0.s4 = as_short(slm_ptr[4*SLM_STRIDE]);
            a0.s5 = as_short(slm_ptr[5*SLM_STRIDE]);
            a0.s6 = as_short(slm_ptr[6*SLM_STRIDE]);
            a0.s7 = as_short(slm_ptr[7*SLM_STRIDE]);

            a1.s0 = as_short(slm_ptr[8*SLM_STRIDE]);
            a1.s1 = as_short(slm_ptr[9*SLM_STRIDE]);
            a1.s2 = as_short(slm_ptr[10*SLM_STRIDE]);
            a1.s3 = as_short(slm_ptr[11*SLM_STRIDE]);
            a1.s4 = as_short(slm_ptr[12*SLM_STRIDE]);
            a1.s5 = as_short(slm_ptr[13*SLM_STRIDE]);
            a1.s6 = as_short(slm_ptr[14*SLM_STRIDE]);
            a1.s7 = as_short(slm_ptr[15*SLM_STRIDE]);

            a2.s0 = as_short(slm_ptr[16*SLM_STRIDE]);
            a2.s1 = as_short(slm_ptr[17*SLM_STRIDE]);
            a2.s2 = as_short(slm_ptr[18*SLM_STRIDE]);
            a2.s3 = as_short(slm_ptr[19*SLM_STRIDE]);
            a2.s4 = as_short(slm_ptr[20*SLM_STRIDE]);
            a2.s5 = as_short(slm_ptr[21*SLM_STRIDE]);
            a2.s6 = as_short(slm_ptr[22*SLM_STRIDE]);
            a2.s7 = as_short(slm_ptr[23*SLM_STRIDE]);

            a3.s0 = as_short(slm_ptr[24*SLM_STRIDE]);
            a3.s1 = as_short(slm_ptr[25*SLM_STRIDE]);
            a3.s2 = as_short(slm_ptr[26*SLM_STRIDE]);
            a3.s3 = as_short(slm_ptr[27*SLM_STRIDE]);
            a3.s4 = as_short(slm_ptr[28*SLM_STRIDE]);
            a3.s5 = as_short(slm_ptr[29*SLM_STRIDE]);
            a3.s6 = as_short(slm_ptr[30*SLM_STRIDE]);
            a3.s7 = as_short(slm_ptr[31*SLM_STRIDE]);

            // Load B for k+16..k+31
            int8 b_reg;
            __global const half* bp = b_base_ptr + 16 * N;

            half h0, h1;
            h0 = bp[0]; h1 = bp[N];
            b_reg.s0 = as_int((short2)(as_short(h0), as_short(h1)));
            h0 = bp[2*N]; h1 = bp[3*N];
            b_reg.s1 = as_int((short2)(as_short(h0), as_short(h1)));
            h0 = bp[4*N]; h1 = bp[5*N];
            b_reg.s2 = as_int((short2)(as_short(h0), as_short(h1)));
            h0 = bp[6*N]; h1 = bp[7*N];
            b_reg.s3 = as_int((short2)(as_short(h0), as_short(h1)));
            h0 = bp[8*N]; h1 = bp[9*N];
            b_reg.s4 = as_int((short2)(as_short(h0), as_short(h1)));
            h0 = bp[10*N]; h1 = bp[11*N];
            b_reg.s5 = as_int((short2)(as_short(h0), as_short(h1)));
            h0 = bp[12*N]; h1 = bp[13*N];
            b_reg.s6 = as_int((short2)(as_short(h0), as_short(h1)));
            h0 = bp[14*N]; h1 = bp[15*N];
            b_reg.s7 = as_int((short2)(as_short(h0), as_short(h1)));

            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b_reg, acc0);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b_reg, acc1);
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b_reg, acc2);
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b_reg, acc3);
        }

        // === Load next A tile into other SLM buffer ===
        if (next_k < K) {
            int next_buf = 1 - buf;
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
    // acc[r] at lane sg_lid = C[wg_row + row_offset + r][sg_col + sg_lid]
    __global half* c_ptr = C + wg_row * N + sg_col + sg_lid;

    // Use intel_sub_group_block_write_us for vectorized stores where possible
    // But C rows are N apart, so we store one element per row per lane

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
// GEMM: C[M,N] = A[M,K] x B[K,N], all half, f32 accumulation
// Launch: GWS = (ceil(N/64)*64, ceil(M/32)), LWS = (64, 1)
// Subgroup size = 16, 4 subgroups per WG
// Tile: 32x64x32, A in SLM, B from global, DPAS f16xf16->f32
// Hardware: Intel B580 (Xe2-HPG), 20 Xe-cores, 96 TFLOPS FP16 XMX

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
    // Tile dimensions
    #define TILE_M 32
    #define TILE_N 64
    #define TILE_K 32
    // SLM stride with padding to reduce bank conflicts
    // A tile in SLM: 32 rows x 32 cols of f16, stride = 32 + 4 = 36 (in f16 elements)
    #define SLM_A_STRIDE 36

    // Workgroup position
    const int wg_n = get_group_id(0); // which 64-col tile
    const int wg_m = get_group_id(1); // which 32-row tile

    const int lid = get_local_id(0);  // 0..63
    const int sg_id = lid / 16;       // subgroup id: 0..3
    const int sg_lid = lid % 16;      // lane within subgroup: 0..15

    // Base positions in output
    const int base_m = wg_m * TILE_M;
    const int base_n = wg_n * TILE_N;

    // Bounds check for entire WG tile
    if (base_m >= M || base_n >= N) return;

    // SLM for A tile: 32 rows x SLM_A_STRIDE half elements (double buffered)
    __local half slm_a[2 * TILE_M * SLM_A_STRIDE];

    // Accumulators: each subgroup computes 8 rows x 64 cols
    // 4 DPAS calls across N (each produces 8x16), so 4 x float8
    float8 acc0 = (float8)(0.0f);
    float8 acc1 = (float8)(0.0f);
    float8 acc2 = (float8)(0.0f);
    float8 acc3 = (float8)(0.0f);

    // Each subgroup handles 8 rows of M
    // sg_id 0 -> rows 0-7, sg_id 1 -> rows 8-15, sg_id 2 -> rows 16-23, sg_id 3 -> rows 24-31
    const int my_row_offset = sg_id * 8;

    // Cooperative A load: 64 WIs load 32x32 = 1024 f16 elements
    // Each WI loads 1024/64 = 16 elements
    // Strategy: each WI loads one row's worth partially
    // 32 rows x 32 cols: assign 2 WIs per row, each loads 16 elements
    const int a_load_row = lid / 2;       // 0..31
    const int a_load_col_base = (lid % 2) * 16; // 0 or 16

    // K-loop with 2x unroll (step = 64 per iteration)
    int buf = 0;

    // Preload first A tile into SLM buffer 0
    {
        int k_offset = 0;
        int a_row = base_m + a_load_row;
        if (a_row < M) {
            __global const half* a_ptr = A + a_row * K + k_offset + a_load_col_base;
            __local half* slm_ptr = slm_a + a_load_row * SLM_A_STRIDE + a_load_col_base;
            // Load 16 half elements
            for (int i = 0; i < 16; i++) {
                slm_ptr[i] = a_ptr[i];
            }
        } else {
            __local half* slm_ptr = slm_a + a_load_row * SLM_A_STRIDE + a_load_col_base;
            for (int i = 0; i < 16; i++) {
                slm_ptr[i] = (half)0.0f;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const int k_iterations = K / TILE_K;

    for (int ki = 0; ki < k_iterations; ki++) {
        int k_offset = ki * TILE_K;
        int next_k_offset = k_offset + TILE_K;

        // Start loading next A tile into other buffer (if not last iteration)
        // We'll do this after DPAS to overlap

        // Load B tile from global: each subgroup needs B[k_offset:k_offset+32, base_n + sg_col*16 : +16]
        // For DPAS: B needs VNNI format: pairs of k packed into int (2 halfs per int)
        // int8 b means 8 ints = 16 half values across k, for 16 columns (one per lane)
        // For k16: int8 b = 16 k-values packed as 8 ints (2 halfs each), distributed across 16 lanes

        // B layout in memory: row-major [K, N]
        // For DPAS k16: we need B[k:k+16, col] for each lane's column
        // int8 per lane: 8 ints, each int = 2 consecutive k-values for that lane's column
        // So b[i] = (B[k+2i, col] | B[k+2i+1, col] << 16) -- VNNI packing

        // Each subgroup processes 4 chunks of 16 columns
        // Subgroup sg_id handles rows [my_row_offset, my_row_offset+8)
        // All subgroups need all 4 N-chunks of B

        // Load B for first k16 block (k_offset to k_offset+15)
        int8 b00, b01, b02, b03; // B for N-chunks 0,1,2,3, first k16
        int8 b10, b11, b12, b13; // B for N-chunks 0,1,2,3, second k16

        // B column for this lane in each N-chunk
        // N-chunk j: columns [base_n + j*16, base_n + j*16 + 15]
        // Lane sg_lid handles column base_n + j*16 + sg_lid

        // Load B chunk 0 (columns base_n+0..15), k16 block 0
        {
            __global const half* b_base = B + k_offset * N + base_n + sg_lid;
            int8 bv;
            for (int i = 0; i < 8; i++) {
                half v0 = b_base[(2*i) * N];
                half v1 = b_base[(2*i+1) * N];
                bv[i] = as_int((short2)(as_short(v0), as_short(v1)));
            }
            b00 = bv;
        }
        {
            __global const half* b_base = B + k_offset * N + base_n + 16 + sg_lid;
            int8 bv;
            for (int i = 0; i < 8; i++) {
                half v0 = b_base[(2*i) * N];
                half v1 = b_base[(2*i+1) * N];
                bv[i] = as_int((short2)(as_short(v0), as_short(v1)));
            }
            b01 = bv;
        }
        {
            __global const half* b_base = B + k_offset * N + base_n + 32 + sg_lid;
            int8 bv;
            for (int i = 0; i < 8; i++) {
                half v0 = b_base[(2*i) * N];
                half v1 = b_base[(2*i+1) * N];
                bv[i] = as_int((short2)(as_short(v0), as_short(v1)));
            }
            b02 = bv;
        }
        {
            __global const half* b_base = B + k_offset * N + base_n + 48 + sg_lid;
            int8 bv;
            for (int i = 0; i < 8; i++) {
                half v0 = b_base[(2*i) * N];
                half v1 = b_base[(2*i+1) * N];
                bv[i] = as_int((short2)(as_short(v0), as_short(v1)));
            }
            b03 = bv;
        }

        // Load B for second k16 block (k_offset+16 to k_offset+31)
        {
            __global const half* b_base = B + (k_offset+16) * N + base_n + sg_lid;
            int8 bv;
            for (int i = 0; i < 8; i++) {
                half v0 = b_base[(2*i) * N];
                half v1 = b_base[(2*i+1) * N];
                bv[i] = as_int((short2)(as_short(v0), as_short(v1)));
            }
            b10 = bv;
        }
        {
            __global const half* b_base = B + (k_offset+16) * N + base_n + 16 + sg_lid;
            int8 bv;
            for (int i = 0; i < 8; i++) {
                half v0 = b_base[(2*i) * N];
                half v1 = b_base[(2*i+1) * N];
                bv[i] = as_int((short2)(as_short(v0), as_short(v1)));
            }
            b11 = bv;
        }
        {
            __global const half* b_base = B + (k_offset+16) * N + base_n + 32 + sg_lid;
            int8 bv;
            for (int i = 0; i < 8; i++) {
                half v0 = b_base[(2*i) * N];
                half v1 = b_base[(2*i+1) * N];
                bv[i] = as_int((short2)(as_short(v0), as_short(v1)));
            }
            b12 = bv;
        }
        {
            __global const half* b_base = B + (k_offset+16) * N + base_n + 48 + sg_lid;
            int8 bv;
            for (int i = 0; i < 8; i++) {
                half v0 = b_base[(2*i) * N];
                half v1 = b_base[(2*i+1) * N];
                bv[i] = as_int((short2)(as_short(v0), as_short(v1)));
            }
            b13 = bv;
        }

        // Load A from SLM for this subgroup
        // Need A[my_row_offset:my_row_offset+8, 0:32] from SLM in DPAS format
        // DPAS a operand: short8 per WI
        // For repcount=8, depth=16: a is 8 rows x 16 k-values
        // short8: each short = 1 f16 value? No...
        // Actually for f16 DPAS: short8 a means 8 shorts per lane
        // With exec_size=16 and repcount=8, depth=16 (k16):
        // A matrix is 8x16 (rows x k), distributed across 16 lanes
        // Each lane holds: 8 shorts = one column of the 8x16 A sub-tile? No...
        // 
        // Actually: for intel_sub_group_f16_f16_matrix_mad_k16:
        // a = short8: 8 rows, each row has 16 f16 values distributed across 16 lanes
        //   So lane i holds a[row][i] for all 8 rows -> short8 = 8 values, one per row
        //   Wait no, that's only 8 values but we need 16 k-values per row...
        //
        // Let me reconsider. The DPAS signature:
        // intel_sub_group_f16_f16_matrix_mad_k16(short8 a, int8 b, float8 acc)
        // - Result: float8 = 8 rows of results, one per lane gives 8x16 output
        // - a: short8 = 8x16 matrix of f16 (8 rows, 16 k-elements)
        //   Distributed: each lane holds 8 shorts. With 16 lanes, total = 128 shorts = 128 f16
        //   That's 8 rows x 16 k = 128 elements. So lane j holds a[0..7][j] -- one k-column
        //   Wait, that's only 8 per lane for 8 rows x 1 k-element each.
        //   Actually no: short8 = 8 shorts per lane, 16 lanes = 128 shorts total
        //   8 rows x 16 k = 128. Distribution: lane j holds k-element j for all 8 rows
        //   So a_lane[i] = A[row_i][k_j] where j = sg_lid... No that doesn't work for k16.
        //
        // Standard Xe2 DPAS layout for f16:
        // a (src1): repcount=8, each rep needs depth/2 = 8 shorts (since depth=16, 2 f16 per int... no)
        // Actually for f16: systolic depth = 16 means 16 f16 MACs per element
        // a is packed: short8 means 8 registers of short (16-bit), across subgroup
        // 
        // The correct interpretation for Xe2 (exec_size=16):
        // short8 a: 8 GRF-rows, each holding 16 shorts (one per lane)
        //   Total: 8*16 = 128 f16 values = 8 rows x 16 k-elements
        //   Layout: a[r] (the r-th short in each lane's short8) = A[row_r][k_lane]
        //   So lane L holds: a[r] = A[base_row + r][k_base + L] for r=0..7
        //
        // int8 b: 8 GRF-rows of int (32-bit), each holding 16 ints (one per lane)  
        //   Total: 8*16 = 128 ints = 256 f16 values = 16 k-elements x 16 columns
        //   Layout: VNNI packed, b[i] at lane L = pack(B[k_base+2i][col_L], B[k_base+2i+1][col_L])
        //
        // float8 result: 8 rows x 16 columns
        //   result[r] at lane L = dot(A[row_r][k_base:k_base+16], B[k_base:k_base+16][col_L])

        // Load A from SLM for first k16 (k=0..15 within tile)
        // Lane sg_lid holds k-element sg_lid for rows my_row_offset..my_row_offset+7
        short8 a0; // first k16
        {
            __local const half* a_slm = slm_a + buf * TILE_M * SLM_A_STRIDE;
            #pragma unroll
            for (int r = 0; r < 8; r++) {
                a0[r] = as_short(a_slm[(my_row_offset + r) * SLM_A_STRIDE + sg_lid]);
            }
        }

        // Load A from SLM for second k16 (k=16..31 within tile)
        short8 a1;
        {
            __local const half* a_slm = slm_a + buf * TILE_M * SLM_A_STRIDE;
            #pragma unroll
            for (int r = 0; r < 8; r++) {
                a1[r] = as_short(a_slm[(my_row_offset + r) * SLM_A_STRIDE + 16 + sg_lid]);
            }
        }

        // DPAS: first k16
        acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b00, acc0);
        acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b01, acc1);
        acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b02, acc2);
        acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b03, acc3);

        // DPAS: second k16
        acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b10, acc0);
        acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b11, acc1);
        acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b12, acc2);
        acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b13, acc3);

        // Load next A tile into other SLM buffer
        if (ki < k_iterations - 1) {
            int next_buf = 1 - buf;
            int a_row = base_m + a_load_row;
            __local half* slm_ptr = slm_a + next_buf * TILE_M * SLM_A_STRIDE + a_load_row * SLM_A_STRIDE + a_load_col_base;
            if (a_row < M) {
                __global const half* a_ptr = A + a_row * K + next_k_offset + a_load_col_base;
                #pragma unroll
                for (int i = 0; i < 16; i++) {
                    slm_ptr[i] = a_ptr[i];
                }
            } else {
                #pragma unroll
                for (int i = 0; i < 16; i++) {
                    slm_ptr[i] = (half)0.0f;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            buf = next_buf;
        }
    }

    // Write C: each subgroup writes 8 rows x 64 cols
    // acc0: rows [my_row_offset..+7], cols [base_n..base_n+15]
    // acc1: rows [my_row_offset..+7], cols [base_n+16..base_n+31]
    // acc2: rows [my_row_offset..+7], cols [base_n+32..base_n+47]
    // acc3: rows [my_row_offset..+7], cols [base_n+48..base_n+63]

    #pragma unroll
    for (int r = 0; r < 8; r++) {
        int row = base_m + my_row_offset + r;
        if (row < M) {
            int col0 = base_n + sg_lid;
            int col1 = base_n + 16 + sg_lid;
            int col2 = base_n + 32 + sg_lid;
            int col3 = base_n + 48 + sg_lid;

            if (col0 < N) C[row * N + col0] = convert_half(acc0[r]);
            if (col1 < N) C[row * N + col1] = convert_half(acc1[r]);
            if (col2 < N) C[row * N + col2] = convert_half(acc2[r]);
            if (col3 < N) C[row * N + col3] = convert_half(acc3[r]);
        }
    }
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

self = <task.TestMatmulOCL object at 0x7ebfd32825a0>
kernel = <function initialize_matmul_kernel.<locals>.run at 0x7ebf6a1351c0>
ocl_queue = <pyopencl._cl.CommandQueue object at 0x7ebf78336020>, _run = 0

    @pytest.mark.parametrize("_run", range(2))
    def test_correctness_wrt_pytorch(self, kernel, ocl_queue, _run):
        args, expected = get_data_on_device(DEFAULT_SHAPE, ocl_queue, seed=_run)
>       kernel(*args)

task.py:236: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
task.py:197: in run
    gws = tuple(_eval_gws_expr(e, env) for e in gws_exprs)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
task.py:197: in <genexpr>
    gws = tuple(_eval_gws_expr(e, env) for e in gws_exprs)
                ^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

expr = 'ceil(N/64)*64', env = {'K': 2560, 'M': 2048, 'N': 2048}

    def _eval_gws_expr(expr: str, env: dict[str, int]) -> int:
        """Evaluate a simple GWS expression like '(N/64)*64' or 'M/32' given M,K,N values."""
        expr = expr.strip()
        allowed = set("0123456789+-*/() MKN")
        if not all(c in allowed for c in expr):
>           raise ValueError(f"Unsafe GWS expression: {expr}")
E           ValueError: Unsafe GWS expression: ceil(N/64)*64

task.py:43: ValueError
________________ TestMatmulOCL.test_correctness_wrt_pytorch[1] _________________

self = <task.TestMatmulOCL object at 0x7ebfd01a7380>
kernel = <function initialize_matmul_kernel.<locals>.run at 0x7ebf6a1351c0>
ocl_queue = <pyopencl._cl.CommandQueue object at 0x7ebf78336020>, _run = 1

    @pytest.mark.parametrize("_run", range(2))
    def test_correctness_wrt_pytorch(self, kernel, ocl_queue, _run):
        args, expected = get_data_on_device(DEFAULT_SHAPE, ocl_queue, seed=_run)
>       kernel(*args)

task.py:236: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
task.py:197: in run
    gws = tuple(_eval_gws_expr(e, env) for e in gws_exprs)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
task.py:197: in <genexpr>
    gws = tuple(_eval_gws_expr(e, env) for e in gws_exprs)
                ^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

expr = 'ceil(N/64)*64', env = {'K': 2560, 'M': 2048, 'N': 2048}

    def _eval_gws_expr(expr: str, env: dict[str, int]) -> int:
        """Evaluate a simple GWS expression like '(N/64)*64' or 'M/32' given M,K,N values."""
        expr = expr.strip()
        allowed = set("0123456789+-*/() MKN")
        if not all(c in allowed for c in expr):
>           raise ValueError(f"Unsafe GWS expression: {expr}")
E           ValueError: Unsafe GWS expression: ceil(N/64)*64

task.py:43: ValueError
_______________ TestMatmulOCL.test_correctness_wrt_reference[0] ________________

self = <task.TestMatmulOCL object at 0x7ebf78323680>
kernel = <function initialize_matmul_kernel.<locals>.run at 0x7ebf6a1351c0>
ocl_queue = <pyopencl._cl.CommandQueue object at 0x7ebf78336020>, _run = 0

    @pytest.mark.parametrize("_run", range(2))
    def test_correctness_wrt_reference(self, kernel, ocl_queue, _run):
        args, _ = get_data_on_device(DEFAULT_SHAPE, ocl_queue, seed=100 + _run)
>       kernel(*args)

task.py:249: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
task.py:197: in run
    gws = tuple(_eval_gws_expr(e, env) for e in gws_exprs)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
task.py:197: in <genexpr>
    gws = tuple(_eval_gws_expr(e, env) for e in gws_exprs)
                ^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

expr = 'ceil(N/64)*64', env = {'K': 2560, 'M': 2048, 'N': 2048}

    def _eval_gws_expr(expr: str, env: dict[str, int]) -> int:
        """Evaluate a simple GWS expression like '(N/64)*64' or 'M/32' given M,K,N values."""
        expr = expr.strip()
        allowed = set("0123456789+-*/() MKN")
        if not all(c in allowed for c in expr):
>           raise ValueError(f"Unsafe GWS expression: {expr}")
E           ValueError: Unsafe GWS expression: ceil(N/64)*64

task.py:43: ValueError
_______________ TestMatmulOCL.test_correctness_wrt_reference[1] ________________

self = <task.TestMatmulOCL object at 0x7ebf78323860>
kernel = <function initialize_matmul_kernel.<locals>.run at 0x7ebf6a1351c0>
ocl_queue = <pyopencl._cl.CommandQueue object at 0x7ebf78336020>, _run = 1

    @pytest.mark.parametrize("_run", range(2))
    def test_correctness_wrt_reference(self, kernel, ocl_queue, _run):
        args, _ = get_data_on_device(DEFAULT_SHAPE, ocl_queue, seed=100 + _run)
>       kernel(*args)

task.py:249: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
task.py:197: in run
    gws = tuple(_eval_gws_expr(e, env) for e in gws_exprs)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
task.py:197: in <genexpr>
    gws = tuple(_eval_gws_expr(e, env) for e in gws_exprs)
                ^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

expr = 'ceil(N/64)*64', env = {'K': 2560, 'M': 2048, 'N': 2048}

    def _eval_gws_expr(expr: str, env: dict[str, int]) -> int:
        """Evaluate a simple GWS expression like '(N/64)*64' or 'M/32' given M,K,N values."""
        expr = expr.strip()
        allowed = set("0123456789+-*/() MKN")
        if not all(c in allowed for c in expr):
>           raise ValueError(f"Unsafe GWS expression: {expr}")
E           ValueError: Unsafe GWS expression: ceil(N/64)*64

task.py:43: ValueError
=============================== warnings summary ===============================
task.py::TestMatmulOCL::test_correctness_wrt_pytorch[0]
  /home/openvino-ci-74/miniforge3/envs/kernel_intel/lib/python3.12/site-packages/pyopencl/cache.py:517: CompilerWarning: Non-empty compiler output encountered. Set the environment variable PYOPENCL_COMPILER_OUTPUT=1 to see more.
    _create_built_program_from_source_cached(

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ============================
FAILED task.py::TestMatmulOCL::test_correctness_wrt_pytorch[0] - ValueError: ...
FAILED task.py::TestMatmulOCL::test_correctness_wrt_pytorch[1] - ValueError: ...
FAILED task.py::TestMatmulOCL::test_correctness_wrt_reference[0] - ValueError...
FAILED task.py::TestMatmulOCL::test_correctness_wrt_reference[1] - ValueError...
================== 4 failed, 1 deselected, 1 warning in 0.77s ==================

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

- **Blocked/Tiled Algorithms**: Process input in blocks to bound peak memory. Trade recomputation for memory savings (e.g., Flash-Attention style). Maintain running accumulators across blocks with proper rescaling.
- **Work-Group Reductions**: Replace atomic operations with O(log N) tree-based reductions in local memory. Synchronize with `group_barrier()` between iterations.