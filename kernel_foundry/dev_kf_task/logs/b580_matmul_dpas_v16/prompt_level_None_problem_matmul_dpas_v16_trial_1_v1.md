

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

### Version 1 (Result: Correctness score: 3 / 5 (compiled but incorrect), runtime in ms: -1.000):
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

### Version 2 (Result: Correctness score: 3 / 5 (compiled but incorrect), runtime in ms: -1.000):
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

### Previous kernel and evaluation:
Here is the last OCL kernel we tested (Result: Correctness score: 5 / 5 (compiles and correct), runtime in ms: 2.560):
```OCL
// GEMM: C[M,N] = A[M,K] x B[K,N], all half precision, f32 accumulation
// Architecture: 64 WIs (4 SGs of 16), TILE 32x64x32, A in SLM double-buffered, B from global
// Launch: LWS=(16,4,1), GWS=(N/4, M/8, 1) i.e. (N/64*16, M/32*4, 1)
// Subgroup size: 16
// For M=N=K=4096: GWS=(1024,512,1), LWS=(16,4,1)

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void matmul(
    __global const half* A,
    __global const half* B,
    __global half* C,
    const int M,
    const int K,
    const int N)
{
    // Work-group tile: 32 rows x 64 cols
    // 4 subgroups, each handles 32 rows x 16 cols
    const int sg_id = get_local_id(1);       // 0..3 (which subgroup)
    const int sg_lid = get_sub_group_local_id(); // 0..15 (lane within SG)

    // WG position in output
    const int wg_row = (get_group_id(1)) * 32;  // M tile start
    const int wg_col = (get_group_id(0)) * 64;  // N tile start

    // Each subgroup handles 16 consecutive columns
    const int sg_col = wg_col + sg_id * 16;

    // SLM for A tile: double-buffered, 32 rows x 32 cols (half)
    // Use stride of 34 to avoid bank conflicts (34 halfs = 68 bytes per row)
    #define SLM_STRIDE 34
    __local half slm_A[2 * 32 * SLM_STRIDE];  // double buffer

    // Accumulators: 32 rows x 16 cols per subgroup = 4 DPAS results of float8
    float8 acc0 = 0.0f;  // rows 0-7
    float8 acc1 = 0.0f;  // rows 8-15
    float8 acc2 = 0.0f;  // rows 16-23
    float8 acc3 = 0.0f;  // rows 24-31

    // Linear local ID for cooperative loading
    const int lid = sg_id * 16 + sg_lid;  // 0..63

    // Preload first A tile into SLM buffer 0
    // A tile: 32 rows x 32 cols = 1024 halfs, 64 WIs load 16 halfs each
    // Each WI loads one row-segment: lid maps to (row, col_chunk)
    // 64 WIs, 32 rows x 32 cols = 1024 elements, 16 per WI
    {
        // Each WI loads 16 consecutive halfs
        // Map: lid/2 = row (0..31), (lid%2)*16 = col offset (0 or 16)
        int a_row = lid / 2;
        int a_col_off = (lid % 2) * 16;
        int a_global_offset = (wg_row + a_row) * K + a_col_off;

        __global const half* a_ptr = A + a_global_offset;

        // Load 16 halfs (use vload8 x2 for efficiency)
        half8 v0 = vload8(0, a_ptr);
        half8 v1 = vload8(1, a_ptr);

        // Store to SLM buffer 0
        int slm_offset = a_row * SLM_STRIDE + a_col_off;
        vstore8(v0, 0, &slm_A[slm_offset]);
        vstore8(v1, 0, &slm_A[slm_offset + 8]);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    int buf = 0;  // current buffer being consumed

    // Main K-loop, step by 32
    for (int k = 0; k < K; k += 32) {
        int next_k = k + 32;

        // Start loading next A tile into other buffer (if not last iteration)
        // We'll do compute first, then load next (with barrier management)

        // === COMPUTE from current SLM buffer ===
        // A is 32x32 in SLM, we process in two K16 steps

        // B pointer for this K-step: B[k:k+32, sg_col:sg_col+16]
        __global const half* b_ptr = B + k * N + sg_col;

        // --- K16 step 0 (k_offset = 0..15) ---
        {
            // Load B[k:k+16, sg_col:sg_col+16] into registers
            // Each lane reads one column, 16 rows = int8 (16 halfs packed as 8 ints)
            int8 b_reg;
            // B is row-major: B[row, col] = B[row*N + col]
            // For DPAS, B needs to be in VNNI format: pairs of k packed into 32-bit
            // int8 b means 8 ints = 16 halfs = 16 k-values for one column
            // Each subgroup lane handles one column (sg_lid = column within 16)
            // We need B in format: b[i] = (B[2i, lane], B[2i+1, lane]) packed as int

            __global const int* b_int_ptr = (__global const int*)(b_ptr + sg_lid);
            b_reg.s0 = *(b_int_ptr); b_int_ptr = (__global const int*)(((__global const half*)b_int_ptr) + N);
            b_reg.s1 = *(__global const int*)(b_ptr + 2*N + sg_lid);  

            // Actually, for VNNI format we need consecutive k pairs in 32-bit lanes
            // B[k, col] with k=0..15, col=sg_lid
            // Packed as int: (B[2i, col], B[2i+1, col]) for i=0..7

            half b_vals[16];
            for (int i = 0; i < 16; i++) {
                b_vals[i] = b_ptr[i * N + sg_lid];
            }
            // Pack into int8 (VNNI format)
            int* b_int = (int*)&b_reg;
            for (int i = 0; i < 8; i++) {
                short h0 = as_short(b_vals[2*i]);
                short h1 = as_short(b_vals[2*i+1]);
                b_int[i] = (int)((uint)(ushort)h0 | ((uint)(ushort)h1 << 16));
            }

            // Load A from SLM for rows 0-7, 8-15, 16-23, 24-31 (k=0..15)
            // A in SLM: slm_A[buf*32*SLM_STRIDE + row*SLM_STRIDE + k_col]
            // For DPAS A operand (short8): 8 rows x 16 k-values packed
            // short8 a means 8 shorts = 8 k-values for one row? No...
            // Actually for intel_sub_group_f16_f16_matrix_mad_k16:
            //   a is short8: across the subgroup, represents 8 rows x 16 k-values
            //   Each lane holds 8 shorts = 8 k-values, lane i holds k-values for specific pattern
            // The A matrix for DPAS k16: 8 rows x 16 cols
            // Distributed: each of 16 lanes holds one column (k-index), 8 rows = short8

            int slm_base = buf * 32 * SLM_STRIDE;

            // For rows 0-7, each lane (sg_lid) reads column sg_lid of k=0..15
            // A[row, k] = slm_A[slm_base + row*SLM_STRIDE + k]
            // Lane sg_lid reads k=sg_lid for all 8 rows
            short8 a0, a1, a2, a3;

            // rows 0-7
            a0.s0 = as_short(slm_A[slm_base + 0*SLM_STRIDE + sg_lid]);
            a0.s1 = as_short(slm_A[slm_base + 1*SLM_STRIDE + sg_lid]);
            a0.s2 = as_short(slm_A[slm_base + 2*SLM_STRIDE + sg_lid]);
            a0.s3 = as_short(slm_A[slm_base + 3*SLM_STRIDE + sg_lid]);
            a0.s4 = as_short(slm_A[slm_base + 4*SLM_STRIDE + sg_lid]);
            a0.s5 = as_short(slm_A[slm_base + 5*SLM_STRIDE + sg_lid]);
            a0.s6 = as_short(slm_A[slm_base + 6*SLM_STRIDE + sg_lid]);
            a0.s7 = as_short(slm_A[slm_base + 7*SLM_STRIDE + sg_lid]);

            // rows 8-15
            a1.s0 = as_short(slm_A[slm_base + 8*SLM_STRIDE + sg_lid]);
            a1.s1 = as_short(slm_A[slm_base + 9*SLM_STRIDE + sg_lid]);
            a1.s2 = as_short(slm_A[slm_base + 10*SLM_STRIDE + sg_lid]);
            a1.s3 = as_short(slm_A[slm_base + 11*SLM_STRIDE + sg_lid]);
            a1.s4 = as_short(slm_A[slm_base + 12*SLM_STRIDE + sg_lid]);
            a1.s5 = as_short(slm_A[slm_base + 13*SLM_STRIDE + sg_lid]);
            a1.s6 = as_short(slm_A[slm_base + 14*SLM_STRIDE + sg_lid]);
            a1.s7 = as_short(slm_A[slm_base + 15*SLM_STRIDE + sg_lid]);

            // rows 16-23
            a2.s0 = as_short(slm_A[slm_base + 16*SLM_STRIDE + sg_lid]);
            a2.s1 = as_short(slm_A[slm_base + 17*SLM_STRIDE + sg_lid]);
            a2.s2 = as_short(slm_A[slm_base + 18*SLM_STRIDE + sg_lid]);
            a2.s3 = as_short(slm_A[slm_base + 19*SLM_STRIDE + sg_lid]);
            a2.s4 = as_short(slm_A[slm_base + 20*SLM_STRIDE + sg_lid]);
            a2.s5 = as_short(slm_A[slm_base + 21*SLM_STRIDE + sg_lid]);
            a2.s6 = as_short(slm_A[slm_base + 22*SLM_STRIDE + sg_lid]);
            a2.s7 = as_short(slm_A[slm_base + 23*SLM_STRIDE + sg_lid]);

            // rows 24-31
            a3.s0 = as_short(slm_A[slm_base + 24*SLM_STRIDE + sg_lid]);
            a3.s1 = as_short(slm_A[slm_base + 25*SLM_STRIDE + sg_lid]);
            a3.s2 = as_short(slm_A[slm_base + 26*SLM_STRIDE + sg_lid]);
            a3.s3 = as_short(slm_A[slm_base + 27*SLM_STRIDE + sg_lid]);
            a3.s4 = as_short(slm_A[slm_base + 28*SLM_STRIDE + sg_lid]);
            a3.s5 = as_short(slm_A[slm_base + 29*SLM_STRIDE + sg_lid]);
            a3.s6 = as_short(slm_A[slm_base + 30*SLM_STRIDE + sg_lid]);
            a3.s7 = as_short(slm_A[slm_base + 31*SLM_STRIDE + sg_lid]);

            // DPAS calls
            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b_reg, acc0);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b_reg, acc1);
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b_reg, acc2);
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b_reg, acc3);
        }

        // --- K16 step 1 (k_offset = 16..31) ---
        {
            __global const half* b_ptr2 = b_ptr + 16 * N;

            half b_vals[16];
            for (int i = 0; i < 16; i++) {
                b_vals[i] = b_ptr2[i * N + sg_lid];
            }
            int8 b_reg;
            int* b_int = (int*)&b_reg;
            for (int i = 0; i < 8; i++) {
                short h0 = as_short(b_vals[2*i]);
                short h1 = as_short(b_vals[2*i+1]);
                b_int[i] = (int)((uint)(ushort)h0 | ((uint)(ushort)h1 << 16));
            }

            int slm_base = buf * 32 * SLM_STRIDE + 16; // offset by 16 in k

            short8 a0, a1, a2, a3;

            a0.s0 = as_short(slm_A[slm_base + 0*SLM_STRIDE + sg_lid]);
            a0.s1 = as_short(slm_A[slm_base + 1*SLM_STRIDE + sg_lid]);
            a0.s2 = as_short(slm_A[slm_base + 2*SLM_STRIDE + sg_lid]);
            a0.s3 = as_short(slm_A[slm_base + 3*SLM_STRIDE + sg_lid]);
            a0.s4 = as_short(slm_A[slm_base + 4*SLM_STRIDE + sg_lid]);
            a0.s5 = as_short(slm_A[slm_base + 5*SLM_STRIDE + sg_lid]);
            a0.s6 = as_short(slm_A[slm_base + 6*SLM_STRIDE + sg_lid]);
            a0.s7 = as_short(slm_A[slm_base + 7*SLM_STRIDE + sg_lid]);

            a1.s0 = as_short(slm_A[slm_base + 8*SLM_STRIDE + sg_lid]);
            a1.s1 = as_short(slm_A[slm_base + 9*SLM_STRIDE + sg_lid]);
            a1.s2 = as_short(slm_A[slm_base + 10*SLM_STRIDE + sg_lid]);
            a1.s3 = as_short(slm_A[slm_base + 11*SLM_STRIDE + sg_lid]);
            a1.s4 = as_short(slm_A[slm_base + 12*SLM_STRIDE + sg_lid]);
            a1.s5 = as_short(slm_A[slm_base + 13*SLM_STRIDE + sg_lid]);
            a1.s6 = as_short(slm_A[slm_base + 14*SLM_STRIDE + sg_lid]);
            a1.s7 = as_short(slm_A[slm_base + 15*SLM_STRIDE + sg_lid]);

            a2.s0 = as_short(slm_A[slm_base + 16*SLM_STRIDE + sg_lid]);
            a2.s1 = as_short(slm_A[slm_base + 17*SLM_STRIDE + sg_lid]);
            a2.s2 = as_short(slm_A[slm_base + 18*SLM_STRIDE + sg_lid]);
            a2.s3 = as_short(slm_A[slm_base + 19*SLM_STRIDE + sg_lid]);
            a2.s4 = as_short(slm_A[slm_base + 20*SLM_STRIDE + sg_lid]);
            a2.s5 = as_short(slm_A[slm_base + 21*SLM_STRIDE + sg_lid]);
            a2.s6 = as_short(slm_A[slm_base + 22*SLM_STRIDE + sg_lid]);
            a2.s7 = as_short(slm_A[slm_base + 23*SLM_STRIDE + sg_lid]);

            a3.s0 = as_short(slm_A[slm_base + 24*SLM_STRIDE + sg_lid]);
            a3.s1 = as_short(slm_A[slm_base + 25*SLM_STRIDE + sg_lid]);
            a3.s2 = as_short(slm_A[slm_base + 26*SLM_STRIDE + sg_lid]);
            a3.s3 = as_short(slm_A[slm_base + 27*SLM_STRIDE + sg_lid]);
            a3.s4 = as_short(slm_A[slm_base + 28*SLM_STRIDE + sg_lid]);
            a3.s5 = as_short(slm_A[slm_base + 29*SLM_STRIDE + sg_lid]);
            a3.s6 = as_short(slm_A[slm_base + 30*SLM_STRIDE + sg_lid]);
            a3.s7 = as_short(slm_A[slm_base + 31*SLM_STRIDE + sg_lid]);

            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b_reg, acc0);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b_reg, acc1);
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b_reg, acc2);
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b_reg, acc3);
        }

        // === Load next A tile (if not last) ===
        if (next_k < K) {
            int next_buf = 1 - buf;
            int a_row = lid / 2;
            int a_col_off = (lid % 2) * 16;
            int a_global_offset = (wg_row + a_row) * K + next_k + a_col_off;

            __global const half* a_ptr = A + a_global_offset;
            half8 v0 = vload8(0, a_ptr);
            half8 v1 = vload8(1, a_ptr);

            int slm_offset = next_buf * 32 * SLM_STRIDE + a_row * SLM_STRIDE + a_col_off;
            vstore8(v0, 0, &slm_A[slm_offset]);
            vstore8(v1, 0, &slm_A[slm_offset + 8]);

            buf = next_buf;
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // === Store C ===
    // Each subgroup writes 32 rows x 16 cols
    // acc0 = rows 0-7, acc1 = rows 8-15, acc2 = rows 16-23, acc3 = rows 24-31
    // Each float8 has 8 values for 8 rows, one column per lane (sg_lid)

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

Console output from running this kernel:

Test result on platform Intel Corporation Battlemage G21 [Intel Graphics]:
==== test session starts

task.py::TestMatmulOCL::test_correctness_wrt_pytorch[0] PASSED           [ 25%]
task.py::TestMatmulOCL::test_correctness_wrt_pytorch[1] PASSED           [ 50%]
task.py::TestMatmulOCL::test_correctness_wrt_reference[0] PASSED         [ 75%]
task.py::TestMatmulOCL::test_correctness_wrt_reference[1] PASSED         [100%]

======================= 4 passed, 1 deselected in 0.98s ========================
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
1. Avoid Bank Conflicts: Local memory is organized into banks (typically 32 banks). Pad shared arrays to avoid stride conflicts, e.g., __local float tile[TILE_SIZE][TILE_SIZE + 1] for transpose operations. Use sequential access patterns within wavefronts.
2. Use Local Memory Strategically: Use __local memory for data reused across work-items in a work-group. Implement tile-based algorithms where each work-group loads a tile into local memory, processes it, then moves to the next tile. Declare with __local float tile[TILE_SIZE].

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

- **Shared Local Memory (SLM) Tiling**: Cache frequently accessed data in SLM using `group_local_memory_for_overwrite`. Synchronize with `group_barrier()` after writes and before reads. Use 16×16 or 32×32 tiles for float data.
- **Kernel Fusion**: Combine sequential operations (e.g., exp → add → activation) into a single kernel. Eliminate intermediate buffers by computing in registers.
- **Hierarchical Parallelism**: Structure work at three levels - work-groups (tile assignment), sub-groups (sub-tile processing), work-items (register tile). Use sub-group shuffles to share data within sub-group without SLM.

[COMPILE ERROR FEEDBACK FROM PREVIOUS TRIAL]
The following compile errors occurred in the previous iteration. Avoid these mistakes in your implementation:
Branch 2: Traceback (most recent call last):
  File "<python_env_path>/pyopencl/__init__.py", line 473, in __getattr__
    knl = Kernel(self, attr)
          ^^^^^^^^^^^^^^^^^^
pyopencl._cl.LogicError: clCreateKernel failed: INVALID_KERNEL_NAME

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/mnt/river/kernel_foundry/kernelfoundry.intern
[END COMPILE ERROR FEEDBACK]
