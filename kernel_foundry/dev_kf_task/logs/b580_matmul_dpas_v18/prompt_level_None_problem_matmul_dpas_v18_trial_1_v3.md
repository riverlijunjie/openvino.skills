

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
// Optimized FP16 GEMM: C[M,N] = A[M,K] x B[K,N]
// Architecture: 64 WIs (4 SGs of 16), TILE 32x64x32, A in SLM, B from global
// Double-buffered SLM with 2x K-loop unroll
// Launch: GWS = (N/64 * 16, M/32), LWS = (16, 4)
//   i.e., GWS[0] = (N/64)*16, GWS[1] = M/32*4... 
//   Actually: LWS = (64, 1), GWS = ((N/64)*(M/32)*64, 1) -- linear
//   Better: LWS = {64, 1}, GWS = {(N/64)*64, M/32}
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
    // Work-group and subgroup identification
    const int wg_id_n = get_group_id(0);  // which 64-col tile
    const int wg_id_m = get_group_id(1);  // which 32-row tile
    const int sg_id = get_sub_group_id();  // 0..3
    const int sg_lid = get_sub_group_local_id(); // 0..15
    const int local_id = get_local_id(0);  // 0..63

    // Base positions in output matrix
    const int base_m = wg_id_m * 32;
    const int base_n = wg_id_n * 64;

    // Each subgroup handles: 8 rows x 64 cols (via 4 DPAS of 8x16 each)
    // SG layout: sg_id determines which 8 rows (sg0: rows 0-7, sg1: 8-15, sg2: 16-23, sg3: 24-31)
    const int sg_row_offset = sg_id * 8;

    // Accumulators: 4 sets of float8 (one per 16-col chunk across 64 cols)
    float8 acc0 = 0.0f; // cols 0-15
    float8 acc1 = 0.0f; // cols 16-31
    float8 acc2 = 0.0f; // cols 32-47
    float8 acc3 = 0.0f; // cols 48-63

    // SLM for A: double buffer, each 32 rows x 32 cols of half
    // Stride 32 halfs = 64 bytes per row. Add padding to avoid bank conflicts: stride = 36
    // Actually use stride=32 (no padding needed for 32-wide with 16-wide SGs)
    #define SLM_A_STRIDE 32  // elements per row
    #define SLM_A_SIZE (32 * SLM_A_STRIDE)  // 1024 halfs = 2048 bytes per buffer
    __local half slm_a[2 * SLM_A_SIZE];  // double buffer: 4096 bytes total

    // Cooperative A load: 64 WIs load 32x32 = 1024 halfs
    // Each WI loads 1024/64 = 16 halfs = 8 iterations of half2 or one chunk
    // Layout: WI i loads elements [i*16 .. i*16+15] linearly mapped to 32x32 tile
    // Row = (local_id * 16 + elem) / 32, Col = (local_id * 16 + elem) % 32

    // Preload first A tile into buffer 0
    int buf_load = 0;
    int buf_compute = 0;

    // Load A[base_m : base_m+32, 0 : 32] into slm_a[buf_load]
    {
        __global const half* a_ptr = A + base_m * K;
        __local half* slm_dst = slm_a + buf_load * SLM_A_SIZE;
        // Each WI loads 16 consecutive halfs
        int base_offset = local_id * 16;
        int row = base_offset / SLM_A_STRIDE;
        int col = base_offset % SLM_A_STRIDE;
        // Global address: A[(base_m + row) * K + (0 + col)]
        // Load 16 halfs (256 bits = 32 bytes)
        __global const half* src = a_ptr + row * K + col;
        __local half* dst = slm_dst + row * SLM_A_STRIDE + col;
        // Since K >= 32 and col < 32, this is safe
        *(__local half16*)dst = *(__global const half16*)src;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Main K-loop: process 64 K-elements per iteration (2x unroll of 32)
    int k = 0;
    for (; k < K; k += 64) {
        // Process first 32 K-elements from current buffer
        // While computing, load next A tile into other buffer
        buf_compute = (k / 32) & 1;
        buf_load = 1 - buf_compute;

        __local half* slm_compute = slm_a + buf_compute * SLM_A_SIZE;

        // --- COMPUTE phase 1 (k .. k+31) ---
        // Load A from SLM for this subgroup: 8 rows, 32 cols (split into 2 x k16)
        // For DPAS k16: need short8 a (8 rows x 16 cols packed as short8)
        // Each row has 16 halfs = 8 shorts. short8 = one row's worth for k=16.
        // Actually intel_sub_group_f16_f16_matrix_mad_k16(short8 a, int8 b, float8 acc)
        // a: short8 per WI, represents 8 rows x 16 cols (each WI holds one row's 16 halfs as short8)
        // b: int8 per WI, represents 16 rows x 16 cols (VNNI packed: pairs of halfs in int)

        // Load A sub-tile for k=0..15: rows [sg_row_offset..sg_row_offset+7], cols [0..15]
        // Each WI in SG reads row (sg_row_offset + lane/2)? No.
        // DPAS a operand: short8 means 8 shorts = 16 halfs packed.
        // For repcount=8, depth=16 (k16): a is 8x16 halfs distributed across 16 WIs
        // Each WI holds short8 = 8 consecutive pairs from different rows
        // Layout: WI[i] holds a[row][i*1 + ...] -- it's complex, let's use block_read

        // SLM A layout: row-major, 32 halfs per row
        // For subgroup block read of 8 rows x 16 cols:
        // Base = slm_compute + sg_row_offset * SLM_A_STRIDE + k_offset
        // Use intel_sub_group_block_read_us8 to read 8 consecutive ushorts per WI

        // k_offset within the 32-col tile: first half is cols 0-15, second half is cols 16-31
        // For k16 iteration 0: cols 0-15
        // For k16 iteration 1: cols 16-31

        // Read A for first k16 (cols 0-15 of the 32-wide SLM tile)
        __local ushort* slm_a_base = (__local ushort*)(slm_compute + sg_row_offset * SLM_A_STRIDE);

        // intel_sub_group_block_read_us8 reads 8 ushorts per WI from SLM
        // With stride = SLM_A_STRIDE (32 halfs = 32 ushorts), reading 8 rows
        // Each row: 16 consecutive ushorts starting at WI's lane offset
        // This gives us the A operand for DPAS

        short8 a_tile_0, a_tile_1;

        // Manual load of A from SLM for DPAS
        // For DPAS on Xe2 (exec_size=16): a is short8 per WI
        // Represents 8 rows, each row has 16 halfs (one per WI lane)
        // So WI[lane] holds: a[0][lane], a[1][lane], ..., a[7][lane] packed as short8

        // Load A tile 0 (k offset 0, cols 0-15):
        {
            short8 a_val;
            __local const ushort* base = (__local const ushort*)(slm_compute + sg_row_offset * SLM_A_STRIDE);
            a_val.s0 = base[0 * SLM_A_STRIDE + sg_lid];
            a_val.s1 = base[1 * SLM_A_STRIDE + sg_lid];
            a_val.s2 = base[2 * SLM_A_STRIDE + sg_lid];
            a_val.s3 = base[3 * SLM_A_STRIDE + sg_lid];
            a_val.s4 = base[4 * SLM_A_STRIDE + sg_lid];
            a_val.s5 = base[5 * SLM_A_STRIDE + sg_lid];
            a_val.s6 = base[6 * SLM_A_STRIDE + sg_lid];
            a_val.s7 = base[7 * SLM_A_STRIDE + sg_lid];
            a_tile_0 = a_val;
        }

        // Load A tile 1 (k offset 16, cols 16-31):
        {
            short8 a_val;
            __local const ushort* base = (__local const ushort*)(slm_compute + sg_row_offset * SLM_A_STRIDE + 16);
            a_val.s0 = base[0 * SLM_A_STRIDE + sg_lid];
            a_val.s1 = base[1 * SLM_A_STRIDE + sg_lid];
            a_val.s2 = base[2 * SLM_A_STRIDE + sg_lid];
            a_val.s3 = base[3 * SLM_A_STRIDE + sg_lid];
            a_val.s4 = base[4 * SLM_A_STRIDE + sg_lid];
            a_val.s5 = base[5 * SLM_A_STRIDE + sg_lid];
            a_val.s6 = base[6 * SLM_A_STRIDE + sg_lid];
            a_val.s7 = base[7 * SLM_A_STRIDE + sg_lid];
            a_tile_1 = a_val;
        }

        // Load B tiles from global memory
        // B[k:k+16, base_n:base_n+64] for first k16
        // DPAS b operand (int8): 16 rows x 16 cols, VNNI packed (pairs of halfs → int)
        // int8 per WI: WI[lane] holds 8 ints, each int = 2 consecutive k-values for one col
        // For 16x16 tile: 16 k-rows, 16 cols. VNNI: pack k[0,1] into int for each col.
        // So b has 8 ints per WI (8 pairs of k-values), 16 WIs (16 cols)

        // B is row-major: B[k_row, col] = B[k_row * N + col]
        // For VNNI format: b[wi] = { B[0,wi]|B[1,wi], B[2,wi]|B[3,wi], ..., B[14,wi]|B[15,wi] }
        // That's 8 ints per WI, each int packs two consecutive k-row values for that WI's column

        // Load B for cols 0-15 (first 16 cols), k offset = k
        __global const uint* b_ptr = (__global const uint*)(B + k * N + base_n);
        // B is half, so B + k*N + base_n points to B[k, base_n]
        // We need VNNI: pairs of rows packed. B[k, col] and B[k+1, col] packed into uint.
        // But B is stored row-major, not VNNI. We need to load and repack.

        // Alternative: load B as halfs and let DPAS handle non-VNNI? No, DPAS requires VNNI for B.
        // We must load B in VNNI format or repack.

        // For Xe2 DPAS with f16: b operand is int8 in VNNI format
        // VNNI for f16: two f16 values from consecutive K packed into one 32-bit int
        // b[wi].s0 = (B[k+1, col_wi] << 16) | B[k+0, col_wi]  (as uint with two f16)
        // b[wi].s1 = (B[k+3, col_wi] << 16) | B[k+2, col_wi]
        // ... etc for 8 pairs = 16 k-values

        // Load B and pack into VNNI format
        int8 b_tile_00, b_tile_01, b_tile_02, b_tile_03;  // 4 col-chunks for k=0..15
        int8 b_tile_10, b_tile_11, b_tile_12, b_tile_13;  // 4 col-chunks for k=16..31

        // For B cols [base_n + 0..15], k rows [k..k+15]:
        {
            __global const half* b_base = B + k * N + base_n + sg_lid;
            int8 b_val;
            b_val.s0 = *((__global const int*)(B + (k+0) * N + base_n) + sg_lid / 2 + (sg_lid % 2) * 0);
            // This is getting complicated. Let me reconsider.

            // Actually for intel_sub_group_f16_f16_matrix_mad_k16:
            // On Xe2 with exec_size=16:
            // a: short8 - each WI holds 8 shorts, representing 8 rows of A, one element per row
            //    (the 16 WIs together hold 16 k-values per row)
            // b: int8 - each WI holds 8 ints, VNNI packed
            //    int = two consecutive k-values for the same column
            //    16 WIs = 16 columns, 8 ints = 8 pairs = 16 k-values

            // So for b: WI[col] holds pairs (B[k+0,col]|B[k+1,col]), (B[k+2,col]|B[k+3,col]), ...
            // where col = base_n + chunk_offset + sg_lid

            // Load: for each pair i (0..7), load B[k+2i, col] and B[k+2i+1, col]
            // col = base_n + sg_lid (for first 16-col chunk)
            int col = base_n + sg_lid;
            int8 bv;
            #pragma unroll
            for (int p = 0; p < 8; p++) {
                half val0 = B[(k + 2*p) * N + col];
                half val1 = B[(k + 2*p + 1) * N + col];
                bv[p] = as_int((short2)(as_short(val0), as_short(val1)));
            }
            b_tile_00 = bv;
        }
        {
            int col = base_n + 16 + sg_lid;
            int8 bv;
            #pragma unroll
            for (int p = 0; p < 8; p++) {
                half val0 = B[(k + 2*p) * N + col];
                half val1 = B[(k + 2*p + 1) * N + col];
                bv[p] = as_int((short2)(as_short(val0), as_short(val1)));
            }
            b_tile_01 = bv;
        }
        {
            int col = base_n + 32 + sg_lid;
            int8 bv;
            #pragma unroll
            for (int p = 0; p < 8; p++) {
                half val0 = B[(k + 2*p) * N + col];
                half val1 = B[(k + 2*p + 1) * N + col];
                bv[p] = as_int((short2)(as_short(val0), as_short(val1)));
            }
            b_tile_02 = bv;
        }
        {
            int col = base_n + 48 + sg_lid;
            int8 bv;
            #pragma unroll
            for (int p = 0; p < 8; p++) {
                half val0 = B[(k + 2*p) * N + col];
                half val1 = B[(k + 2*p + 1) * N + col];
                bv[p] = as_int((short2)(as_short(val0), as_short(val1)));
            }
            b_tile_03 = bv;
        }

        // DPAS for first k16
        acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a_tile_0, b_tile_00, acc0);
        acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a_tile_0, b_tile_01, acc1);
        acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a_tile_0, b_tile_02, acc2);
        acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a_tile_0, b_tile_03, acc3);

        // Load B for second k16 (k+16..k+31)
        {
            int col = base_n + sg_lid;
            int8 bv;
            #pragma unroll
            for (int p = 0; p < 8; p++) {
                half val0 = B[(k + 16 + 2*p) * N + col];
                half val1 = B[(k + 16 + 2*p + 1) * N + col];
                bv[p] = as_int((short2)(as_short(val0), as_short(val1)));
            }
            b_tile_10 = bv;
        }
        {
            int col = base_n + 16 + sg_lid;
            int8 bv;
            #pragma unroll
            for (int p = 0; p < 8; p++) {
                half val0 = B[(k + 16 + 2*p) * N + col];
                half val1 = B[(k + 16 + 2*p + 1) * N + col];
                bv[p] = as_int((short2)(as_short(val0), as_short(val1)));
            }
            b_tile_11 = bv;
        }
        {
            int col = base_n + 32 + sg_lid;
            int8 bv;
            #pragma unroll
            for (int p = 0; p < 8; p++) {
                half val0 = B[(k + 16 + 2*p) * N + col];
                half val1 = B[(k + 16 + 2*p + 1) * N + col];
                bv[p] = as_int((short2)(as_short(val0), as_short(val1)));
            }
            b_tile_12 = bv;
        }
        {
            int col = base_n + 48 + sg_lid;
            int8 bv;
            #pragma unroll
            for (int p = 0; p < 8; p++) {
                half val0 = B[(k + 16 + 2*p) * N + col];
                half val1 = B[(k + 16 + 2*p + 1) * N + col];
                bv[p] = as_int((short2)(as_short(val0), as_short(val1)));
            }
            b_tile_13 = bv;
        }

        // DPAS for second k16
        acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a_tile_1, b_tile_10, acc0);
        acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a_tile_1, b_tile_11, acc1);
        acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a_tile_1, b_tile_12, acc2);
        acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a_tile_1, b_tile_13, acc3);

        // --- Load next A tile into other SLM buffer (for k+32..k+63) ---
        if (k + 32 < K) {
            __global const half* a_src = A + base_m * K + (k + 32);
            __local half* slm_dst = slm_a + buf_load * SLM_A_SIZE;
            int base_offset = local_id * 16;
            int row = base_offset / SLM_A_STRIDE;
            int col = base_offset % SLM_A_STRIDE;
            __global const half* src = a_src + row * K + col;
            __local half* dst = slm_dst + row * SLM_A_STRIDE + col;
            *(__local half16*)dst = *(__global const half16*)src;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // --- COMPUTE phase 2 (k+32 .. k+63) ---
        if (k + 32 < K) {
            buf_compute = buf_load;

            __local half* slm_compute2 = slm_a + buf_compute * SLM_A_SIZE;

            short8 a2_tile_0, a2_tile_1;
            {
                __local const ushort* base = (__local const ushort*)(slm_compute2 + sg_row_offset * SLM_A_STRIDE);
                short8 a_val;
                a_val.s0 = base[0 * SLM_A_STRIDE + sg_lid];
                a_val.s1 = base[1 * SLM_A_STRIDE + sg_lid];
                a_val.s2 = base[2 * SLM_A_STRIDE + sg_lid];
                a_val.s3 = base[3 * SLM_A_STRIDE + sg_lid];
                a_val.s4 = base[4 * SLM_A_STRIDE + sg_lid];
                a_val.s5 = base[5 * SLM_A_STRIDE + sg_lid];
                a_val.s6 = base[6 * SLM_A_STRIDE + sg_lid];
                a_val.s7 = base[7 * SLM_A_STRIDE + sg_lid];
                a2_tile_0 = a_val;
            }
            {
                __local const ushort* base = (__local const ushort*)(slm_compute2 + sg_row_offset * SLM_A_STRIDE + 16);
                short8 a_val;
                a_val.s0 = base[0 * SLM_A_STRIDE + sg_lid];
                a_val.s1 = base[1 * SLM_A_STRIDE + sg_lid];
                a_val.s2 = base[2 * SLM_A_STRIDE + sg_lid];
                a_val.s3 = base[3 * SLM_A_STRIDE + sg_lid];
                a_val.s4 = base[4 * SLM_A_STRIDE + sg_lid];
                a_val.s5 = base[5 * SLM_A_STRIDE + sg_lid];
                a_val.s6 = base[6 * SLM_A_STRIDE + sg_lid];
                a_val.s7 = base[7 * SLM_A_STRIDE + sg_lid];
                a2_tile_1 = a_val;
            }

            // B for k+32..k+47
            {
                int col = base_n + sg_lid;
                int8 bv;
                #pragma unroll
                for (int p = 0; p < 8; p++) {
                    half val0 = B[(k + 32 + 2*p) * N + col];
                    half val1 = B[(k + 32 + 2*p + 1) * N + col];
                    bv[p] = as_int((short2)(as_short(val0), as_short(val1)));
                }
                b_tile_00 = bv;
            }
            {
                int col = base_n + 16 + sg_lid;
                int8 bv;
                #pragma unroll
                for (int p = 0; p < 8; p++) {
                    half val0 = B[(k + 32 + 2*p) * N + col];
                    half val1 = B[(k + 32 + 2*p + 1) * N + col];
                    bv[p] = as_int((short2)(as_short(val0), as_short(val1)));
                }
                b_tile_01 = bv;
            }
            {
                int col = base_n + 32 + sg_lid;
                int8 bv;
                #pragma unroll
                for (int p = 0; p < 8; p++) {
                    half val0 = B[(k + 32 + 2*p) * N + col];
                    half val1 = B[(k + 32 + 2*p + 1) * N + col];
                    bv[p] = as_int((short2)(as_short(val0), as_short(val1)));
                }
                b_tile_02 = bv;
            }
            {
                int col = base_n + 48 + sg_lid;
                int8 bv;
                #pragma unroll
                for (int p = 0; p < 8; p++) {
                    half val0 = B[(k + 32 + 2*p) * N + col];
                    half val1 = B[(k + 32 + 2*p + 1) * N + col];
                    bv[p] = as_int((short2)(as_short(val0), as_short(val1)));
                }
                b_tile_03 = bv;
            }

            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a2_tile_0, b_tile_00, acc0);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a2_tile_0, b_tile_01, acc1);
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2_tile_0, b_tile_02, acc2);
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a2_tile_0, b_tile_03, acc3);

            // B for k+48..k+63
            {
                int col = base_n + sg_lid;
                int8 bv;
                #pragma unroll
                for (int p = 0; p < 8; p++) {
                    half val0 = B[(k + 48 + 2*p) * N + col];
                    half val1 = B[(k + 48 + 2*p + 1) * N + col];
                    bv[p] = as_int((short2)(as_short(val0), as_short(val1)));
                }
                b_tile_10 = bv;
            }
            {
                int col = base_n + 16 + sg_lid;
                int8 bv;
                #pragma unroll
                for (int p = 0; p < 8; p++) {
                    half val0 = B[(k + 48 + 2*p) * N + col];
                    half val1 = B[(k + 48 + 2*p + 1) * N + col];
                    bv[p] = as_int((short2)(as_short(val0), as_short(val1)));
                }
                b_tile_11 = bv;
            }
            {
                int col = base_n + 32 + sg_lid;
                int8 bv;
                #pragma unroll
                for (int p = 0; p < 8; p++) {
                    half val0 = B[(k + 48 + 2*p) * N + col];
                    half val1 = B[(k + 48 + 2*p + 1) * N + col];
                    bv[p] = as_int((short2)(as_short(val0), as_short(val1)));
                }
                b_tile_12 = bv;
            }
            {
                int col = base_n + 48 + sg_lid;
                int8 bv;
                #pragma unroll
                for (int p = 0; p < 8; p++) {
                    half val0 = B[(k + 48 + 2*p) * N + col];
                    half val1 = B[(k + 48 + 2*p + 1) * N + col];
                    bv[p] = as_int((short2)(as_short(val0), as_short(val1)));
                }
                b_tile_13 = bv;
            }

            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a2_tile_1, b_tile_10, acc0);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a2_tile_1, b_tile_11, acc1);
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2_tile_1, b_tile_12, acc2);
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a2_tile_1, b_tile_13, acc3);
        }

        // Load next iteration's A tile (k+64..k+95) into the first buffer
        if (k + 64 < K) {
            __global const half* a_src = A + base_m * K + (k + 64);
            __local half* slm_dst = slm_a + buf_compute * SLM_A_SIZE;
            int base_offset = local_id * 16;
            int row = base_offset / SLM_A_STRIDE;
            int col = base_offset % SLM_A_STRIDE;
            __global const half* src = a_src + row * K + col;
            __local half* dst = slm_dst + row * SLM_A_STRIDE + col;
            *(__local half16*)dst = *(__global const half16*)src;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store results to C
    // Each subgroup writes 8 rows x 64 cols
    // acc0..acc3 are float8 each: 8 rows x 1 col per WI, 16 WIs = 16 cols
    int c_row = base_m + sg_row_offset;
    int c_col = base_n + sg_lid;

    #pragma unroll
    for (int r = 0; r < 8; r++) {
        if (c_row + r < M) {
            float val0, val1, val2, val3;
            switch(r) {
                case 0: val0=acc0.s0; val1=acc1.s0; val2=acc2.s0; val3=acc3.s0; break;
                case 1: val0=acc0.s1; val1=acc1.s1; val2=acc2.s1; val3=acc3.s1; break;
                case 2: val0=acc0.s2; val1=acc1.s2; val2=acc2.s2; val3=acc3.s2; break;
                case 3: val0=acc0.s3; val1=acc1.s3; val2=acc2.s3; val3=acc3.s3; break;
                case 4: val0=acc0.s4; val1=acc1.s4; val2=acc2.s4; val3=acc3.s4; break;
                case 5: val0=acc0.s5; val1=acc1.s5; val2=acc2.s5; val3=acc3.s5; break;
                case 6: val0=acc0.s6; val1=acc1.s6; val2=acc2.s6; val3=acc3.s6; break;
                case 7: val0=acc0.s7; val1=acc1.s7; val2=acc2.s7; val3=acc3.s7; break;
            }
            C[(c_row + r) * N + c_col] = convert_half(val0);
            C[(c_row + r) * N + c_col + 16] = convert_half(val1);
            C[(c_row + r) * N + c_col + 32] = convert_half(val2);
            C[(c_row + r) * N + c_col + 48] = convert_half(val3);
        }
    }
}
```

### Version 2 (Result: Correctness score: 5 / 5 (compiles and correct), runtime in ms: 1.560):
```OCL
// GEMM: C[M,N] = A[M,K] x B[K,N], all half precision, f32 accumulation
// Launch: GWS = (N/64 * 64, M/32), LWS = (64, 1)
// Subgroup size = 16, 4 subgroups per workgroup
// Tile: 32x64x32 (M x N x K)
// Architecture: A in SLM (double-buffered), B from global/L2

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
    // Tile indices
    const int tile_n = get_group_id(0);  // which 64-col tile
    const int tile_m = get_group_id(1);  // which 32-row tile

    const int lid = get_local_id(0);     // 0..63
    const int sg_id = lid / 16;          // subgroup id 0..3
    const int sg_lid = lid % 16;         // lane within subgroup

    // Base pointers for this tile
    const int base_m = tile_m * 32;
    const int base_n = tile_n * 64;

    // SLM for A: double-buffered, 32 rows x 32 cols with padding
    // Stride = 34 (32 + 2 padding) to avoid bank conflicts
    // Each buffer: 32 * 34 * 2 bytes = 2176 bytes
    // Total: 4352 bytes
    #define SLM_STRIDE 34
    __local half slm_a[2 * 32 * SLM_STRIDE];

    // Accumulators: each subgroup computes 32 rows x 16 cols
    // 4 vertical blocks of 8 rows each
    float8 acc0 = 0.0f;
    float8 acc1 = 0.0f;
    float8 acc2 = 0.0f;
    float8 acc3 = 0.0f;

    // Each subgroup handles columns [sg_id*16 .. sg_id*16+15] of the N tile
    const int b_col_base = base_n + sg_id * 16;

    // Cooperative A load: 64 WIs load 32x32 halfs = 1024 elements
    // Each WI loads 1024/64 = 16 elements
    // Layout: WI i loads elements [i*16 .. i*16+15]
    // Map to row/col: 32 cols per row, so row = (i*16)/32, col_start = (i*16)%32
    // Actually with stride 34 in SLM, we need careful mapping
    // Better: each WI loads a specific pattern
    // 64 WIs, 32 rows, 32 cols -> 2 WIs per row, each loads 16 consecutive halfs

    // Preload first A tile into buffer 0
    int buf_cur = 0;
    {
        // Each WI loads 16 halfs. 64 WIs * 16 = 1024 = 32*32
        // WI lid loads row (lid/2), cols [(lid%2)*16 .. (lid%2)*16+15]
        int a_row = lid / 2;
        int a_col_start = (lid % 2) * 16;
        int a_global_offset = (base_m + a_row) * K + a_col_start;

        __local half* slm_dst = slm_a + a_row * SLM_STRIDE + a_col_start;

        // Load 16 halfs from global A
        __global const half* a_src = A + a_global_offset;
        half16 a_val = vload16(0, a_src);
        vstore16(a_val, 0, slm_dst);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Main K-loop, step by 32, unrolled 2x (step by 64)
    const int k_iters = K / 32;

    for (int ki = 0; ki < k_iters; ki++) {
        int k_offset = ki * 32;
        int next_k = k_offset + 32;
        int next_buf = 1 - buf_cur;

        // Load next A tile into next buffer (if not last iteration)
        if (next_k < K) {
            int a_row = lid / 2;
            int a_col_start = (lid % 2) * 16;
            int a_global_offset = (base_m + a_row) * K + next_k + a_col_start;

            __local half* slm_dst = slm_a + next_buf * 32 * SLM_STRIDE + a_row * SLM_STRIDE + a_col_start;

            __global const half* a_src = A + a_global_offset;
            half16 a_val = vload16(0, a_src);
            vstore16(a_val, 0, slm_dst);
        }

        // Compute with current buffer
        // Load B: 32 rows x 16 cols for this subgroup
        // B is [K, N] row-major. We need B[k_offset..k_offset+31, b_col_base..b_col_base+15]
        // For DPAS: intel_sub_group_f16_f16_matrix_mad_k16(short8 a, int8 b, float8 acc)
        // a (short8): 8 rows of A, each with k16 packed as pairs -> short8 per WI
        //   Actually for subgroup of 16: short8 means 8 elements per WI, 
        //   representing 8 rows × 16 k-elements (each WI holds one k-column pair)
        // b (int8): 16 cols × 16 k-elements packed as int8 per WI
        //   Each WI in the subgroup holds 8 ints = 16 halfs for its column

        // DPAS k16 step 0 (k=0..15 within tile)
        // Load A from SLM for k=0..15
        // A in SLM: 32 rows x 34 stride, we need cols 0..15
        // For DPAS a operand: short8 = 8 rows × (k16 distributed across 16 lanes)
        // Each lane sg_lid gets A[row][sg_lid] for the k-dimension
        // short8 means 8 consecutive rows, one k-element per lane
        // Wait - DPAS k16 means depth=16, so a has 8 rows × 16 k-values
        // Distributed: each of 16 lanes holds 8 values (one per row), and the k-index = lane_id
        // So short8 a: lane sg_lid holds A[row+0..row+7][k_base + sg_lid] packed as short8

        __local half* slm_base = slm_a + buf_cur * 32 * SLM_STRIDE;

        // K-step 0: k=0..15
        short8 a00, a01, a02, a03;
        {
            // For rows 0..7, 8..15, 16..23, 24..31, load A[row][sg_lid] from SLM
            // sg_lid is the k-index (0..15)
            a00 = (short8)(
                as_short(slm_base[0 * SLM_STRIDE + sg_lid]),
                as_short(slm_base[1 * SLM_STRIDE + sg_lid]),
                as_short(slm_base[2 * SLM_STRIDE + sg_lid]),
                as_short(slm_base[3 * SLM_STRIDE + sg_lid]),
                as_short(slm_base[4 * SLM_STRIDE + sg_lid]),
                as_short(slm_base[5 * SLM_STRIDE + sg_lid]),
                as_short(slm_base[6 * SLM_STRIDE + sg_lid]),
                as_short(slm_base[7 * SLM_STRIDE + sg_lid])
            );
            a01 = (short8)(
                as_short(slm_base[8 * SLM_STRIDE + sg_lid]),
                as_short(slm_base[9 * SLM_STRIDE + sg_lid]),
                as_short(slm_base[10 * SLM_STRIDE + sg_lid]),
                as_short(slm_base[11 * SLM_STRIDE + sg_lid]),
                as_short(slm_base[12 * SLM_STRIDE + sg_lid]),
                as_short(slm_base[13 * SLM_STRIDE + sg_lid]),
                as_short(slm_base[14 * SLM_STRIDE + sg_lid]),
                as_short(slm_base[15 * SLM_STRIDE + sg_lid])
            );
            a02 = (short8)(
                as_short(slm_base[16 * SLM_STRIDE + sg_lid]),
                as_short(slm_base[17 * SLM_STRIDE + sg_lid]),
                as_short(slm_base[18 * SLM_STRIDE + sg_lid]),
                as_short(slm_base[19 * SLM_STRIDE + sg_lid]),
                as_short(slm_base[20 * SLM_STRIDE + sg_lid]),
                as_short(slm_base[21 * SLM_STRIDE + sg_lid]),
                as_short(slm_base[22 * SLM_STRIDE + sg_lid]),
                as_short(slm_base[23 * SLM_STRIDE + sg_lid])
            );
            a03 = (short8)(
                as_short(slm_base[24 * SLM_STRIDE + sg_lid]),
                as_short(slm_base[25 * SLM_STRIDE + sg_lid]),
                as_short(slm_base[26 * SLM_STRIDE + sg_lid]),
                as_short(slm_base[27 * SLM_STRIDE + sg_lid]),
                as_short(slm_base[28 * SLM_STRIDE + sg_lid]),
                as_short(slm_base[29 * SLM_STRIDE + sg_lid]),
                as_short(slm_base[30 * SLM_STRIDE + sg_lid]),
                as_short(slm_base[31 * SLM_STRIDE + sg_lid])
            );
        }

        // Load B for k=0..15: B[k_offset + 0..15, b_col_base + 0..15]
        // For DPAS b operand (int8): VNNI format - pairs of k packed into int
        // int8 per WI: 8 ints = 8 pairs of k × 1 column
        // Each lane sg_lid corresponds to one of 16 columns
        // int8 b means: b[i] = pack(B[k_base + 2*i][col], B[k_base + 2*i + 1][col]) as int
        int8 b0;
        {
            __global const half* b_ptr = B + k_offset * N + b_col_base + sg_lid;
            b0 = (int8)(
                as_int((half2)(b_ptr[0 * N], b_ptr[1 * N])),
                as_int((half2)(b_ptr[2 * N], b_ptr[3 * N])),
                as_int((half2)(b_ptr[4 * N], b_ptr[5 * N])),
                as_int((half2)(b_ptr[6 * N], b_ptr[7 * N])),
                as_int((half2)(b_ptr[8 * N], b_ptr[9 * N])),
                as_int((half2)(b_ptr[10 * N], b_ptr[11 * N])),
                as_int((half2)(b_ptr[12 * N], b_ptr[13 * N])),
                as_int((half2)(b_ptr[14 * N], b_ptr[15 * N]))
            );
        }

        // DPAS calls for k=0..15
        acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a00, b0, acc0);
        acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a01, b0, acc1);
        acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a02, b0, acc2);
        acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a03, b0, acc3);

        // K-step 1: k=16..31
        short8 a10, a11, a12, a13;
        {
            // k offset in SLM is +16
            a10 = (short8)(
                as_short(slm_base[0 * SLM_STRIDE + 16 + sg_lid]),
                as_short(slm_base[1 * SLM_STRIDE + 16 + sg_lid]),
                as_short(slm_base[2 * SLM_STRIDE + 16 + sg_lid]),
                as_short(slm_base[3 * SLM_STRIDE + 16 + sg_lid]),
                as_short(slm_base[4 * SLM_STRIDE + 16 + sg_lid]),
                as_short(slm_base[5 * SLM_STRIDE + 16 + sg_lid]),
                as_short(slm_base[6 * SLM_STRIDE + 16 + sg_lid]),
                as_short(slm_base[7 * SLM_STRIDE + 16 + sg_lid])
            );
            a11 = (short8)(
                as_short(slm_base[8 * SLM_STRIDE + 16 + sg_lid]),
                as_short(slm_base[9 * SLM_STRIDE + 16 + sg_lid]),
                as_short(slm_base[10 * SLM_STRIDE + 16 + sg_lid]),
                as_short(slm_base[11 * SLM_STRIDE + 16 + sg_lid]),
                as_short(slm_base[12 * SLM_STRIDE + 16 + sg_lid]),
                as_short(slm_base[13 * SLM_STRIDE + 16 + sg_lid]),
                as_short(slm_base[14 * SLM_STRIDE + 16 + sg_lid]),
                as_short(slm_base[15 * SLM_STRIDE + 16 + sg_lid])
            );
            a12 = (short8)(
                as_short(slm_base[16 * SLM_STRIDE + 16 + sg_lid]),
                as_short(slm_base[17 * SLM_STRIDE + 16 + sg_lid]),
                as_short(slm_base[18 * SLM_STRIDE + 16 + sg_lid]),
                as_short(slm_base[19 * SLM_STRIDE + 16 + sg_lid]),
                as_short(slm_base[20 * SLM_STRIDE + 16 + sg_lid]),
                as_short(slm_base[21 * SLM_STRIDE + 16 + sg_lid]),
                as_short(slm_base[22 * SLM_STRIDE + 16 + sg_lid]),
                as_short(slm_base[23 * SLM_STRIDE + 16 + sg_lid])
            );
            a13 = (short8)(
                as_short(slm_base[24 * SLM_STRIDE + 16 + sg_lid]),
                as_short(slm_base[25 * SLM_STRIDE + 16 + sg_lid]),
                as_short(slm_base[26 * SLM_STRIDE + 16 + sg_lid]),
                as_short(slm_base[27 * SLM_STRIDE + 16 + sg_lid]),
                as_short(slm_base[28 * SLM_STRIDE + 16 + sg_lid]),
                as_short(slm_base[29 * SLM_STRIDE + 16 + sg_lid]),
                as_short(slm_base[30 * SLM_STRIDE + 16 + sg_lid]),
                as_short(slm_base[31 * SLM_STRIDE + 16 + sg_lid])
            );
        }

        // Load B for k=16..31
        int8 b1;
        {
            __global const half* b_ptr = B + (k_offset + 16) * N + b_col_base + sg_lid;
            b1 = (int8)(
                as_int((half2)(b_ptr[0 * N], b_ptr[1 * N])),
                as_int((half2)(b_ptr[2 * N], b_ptr[3 * N])),
                as_int((half2)(b_ptr[4 * N], b_ptr[5 * N])),
                as_int((half2)(b_ptr[6 * N], b_ptr[7 * N])),
                as_int((half2)(b_ptr[8 * N], b_ptr[9 * N])),
                as_int((half2)(b_ptr[10 * N], b_ptr[11 * N])),
                as_int((half2)(b_ptr[12 * N], b_ptr[13 * N])),
                as_int((half2)(b_ptr[14 * N], b_ptr[15 * N]))
            );
        }

        // DPAS calls for k=16..31
        acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a10, b1, acc0);
        acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a11, b1, acc1);
        acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a12, b1, acc2);
        acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a13, b1, acc3);

        // Wait for next A load to complete before swapping buffers
        barrier(CLK_LOCAL_MEM_FENCE);
        buf_cur = next_buf;
    }

    // Store C: each subgroup writes 32 rows x 16 cols
    // acc0 = rows 0..7, acc1 = rows 8..15, acc2 = rows 16..23, acc3 = rows 24..31
    // float8 per WI, lane sg_lid corresponds to column sg_lid within the 16-col block
    // Each element of float8 is one row

    __global half* c_base = C + base_m * N + b_col_base + sg_lid;

    // Store acc0 (rows 0..7)
    c_base[0 * N] = convert_half(acc0.s0);
    c_base[1 * N] = convert_half(acc0.s1);
    c_base[2 * N] = convert_half(acc0.s2);
    c_base[3 * N] = convert_half(acc0.s3);
    c_base[4 * N] = convert_half(acc0.s4);
    c_base[5 * N] = convert_half(acc0.s5);
    c_base[6 * N] = convert_half(acc0.s6);
    c_base[7 * N] = convert_half(acc0.s7);

    // Store acc1 (rows 8..15)
    c_base[8 * N] = convert_half(acc1.s0);
    c_base[9 * N] = convert_half(acc1.s1);
    c_base[10 * N] = convert_half(acc1.s2);
    c_base[11 * N] = convert_half(acc1.s3);
    c_base[12 * N] = convert_half(acc1.s4);
    c_base[13 * N] = convert_half(acc1.s5);
    c_base[14 * N] = convert_half(acc1.s6);
    c_base[15 * N] = convert_half(acc1.s7);

    // Store acc2 (rows 16..23)
    c_base[16 * N] = convert_half(acc2.s0);
    c_base[17 * N] = convert_half(acc2.s1);
    c_base[18 * N] = convert_half(acc2.s2);
    c_base[19 * N] = convert_half(acc2.s3);
    c_base[20 * N] = convert_half(acc2.s4);
    c_base[21 * N] = convert_half(acc2.s5);
    c_base[22 * N] = convert_half(acc2.s6);
    c_base[23 * N] = convert_half(acc2.s7);

    // Store acc3 (rows 24..31)
    c_base[24 * N] = convert_half(acc3.s0);
    c_base[25 * N] = convert_half(acc3.s1);
    c_base[26 * N] = convert_half(acc3.s2);
    c_base[27 * N] = convert_half(acc3.s3);
    c_base[28 * N] = convert_half(acc3.s4);
    c_base[29 * N] = convert_half(acc3.s5);
    c_base[30 * N] = convert_half(acc3.s6);
    c_base[31 * N] = convert_half(acc3.s7);
}
```

### Previous kernel and evaluation:
Here is the last OCL kernel we tested (Result: Correctness score: 5 / 5 (compiles and correct), runtime in ms: 1.510):
```OCL
// GEMM: C[M,N] = A[M,K] x B[K,N], all half, f32 accumulation
// Launch: GWS = ((N/64)*16, (M/32)*4, 1), LWS = (16, 4, 1)
// 4 subgroups of 16 WIs per WG. Each WG computes 32x64 tile.
// Each subgroup computes 32x16 using DPAS.
// A in SLM (double-buffered), B from global/L2.
// TILE_M=32, TILE_N=64, TILE_K=32. K must be divisible by 32.

#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable

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
    const int sg_id = get_local_id(1); // subgroup index 0-3
    const int sg_lane = get_local_id(0); // lane within subgroup 0-15

    const int tile_row = wg_m * 32;
    const int tile_col = wg_n * 64 + sg_id * 16;

    // Linear local ID for cooperative loads
    const int lid = sg_id * 16 + sg_lane; // 0..63

    // SLM for A: double-buffered, 32 rows x 32 cols of half
    // Pad stride to 34 to avoid bank conflicts (34 halfs = 68 bytes per row)
    #define SLM_STRIDE 34
    #define SLM_BUF_SIZE (32 * SLM_STRIDE)  // per buffer in halfs
    __local half slm_a[2 * SLM_BUF_SIZE];

    // Accumulators: 32 rows x 16 cols per subgroup = 4 blocks of 8 rows
    float8 acc0 = 0.0f;
    float8 acc1 = 0.0f;
    float8 acc2 = 0.0f;
    float8 acc3 = 0.0f;

    // Preload first A tile into SLM buffer 0
    // A tile: 32 rows x 32 cols. 64 WIs load 32*32/64 = 16 halfs each
    // Each WI loads one half16 vector (16 consecutive elements)
    // Map: lid covers 64 positions, we need 32*32=1024 halfs = 64*16
    {
        // Each WI loads 16 halfs. lid determines which row/col chunk.
        // 32 cols / 16 = 2 chunks per row, so 64 WIs cover 32 rows * 2 chunks
        int chunk_row = lid / 2;
        int chunk_col = (lid % 2) * 16;
        int a_row = tile_row + chunk_row;
        int a_col = chunk_col; // k=0

        __global const half* a_ptr = A + a_row * K + a_col;
        half16 a_val = vload16(0, a_ptr);

        int slm_offset = chunk_row * SLM_STRIDE + chunk_col;
        vstore16(a_val, 0, &slm_a[slm_offset]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int cur_buf = 0;

    // Main K-loop, step by 32
    for (int k = 0; k < K; k += 32) {
        int next_k = k + 32;
        int next_buf = 1 - cur_buf;

        // Load next A tile into next SLM buffer (if not last iteration)
        if (next_k < K) {
            int chunk_row = lid / 2;
            int chunk_col = (lid % 2) * 16;
            int a_row = tile_row + chunk_row;
            int a_col = next_k + chunk_col;

            __global const half* a_ptr = A + a_row * K + a_col;
            half16 a_val = vload16(0, a_ptr);

            int slm_offset = next_buf * SLM_BUF_SIZE + chunk_row * SLM_STRIDE + chunk_col;
            vstore16(a_val, 0, &slm_a[slm_offset]);
        }

        // Compute with current buffer
        // Load B from global: 32 rows x 16 cols for this subgroup
        // DPAS needs B in packed format: int8 = 16 halfs packed as 8 ints (16 rows x 16 cols, k16)
        // B layout: row-major [K, N], so B[k_offset, tile_col] 
        // For DPAS k16: we process k in two steps of 16

        __local half* cur_slm = &slm_a[cur_buf * SLM_BUF_SIZE];

        // Process k_inner = 0..15 (first k16 block)
        {
            // Load A from SLM for DPAS: need short8 per 8-row block
            // short8 a means 8 rows x 16 cols packed: each short = 1 half
            // Actually for DPAS k16: short8 a = 8 rows, each with k16 packed as short
            // intel_sub_group_f16_f16_matrix_mad_k16(short8 a, int8 b, float8 acc)
            // a: short8 - each lane holds 8 consecutive k-values for its row? No.
            // a: short8 across subgroup: 16 lanes × short8 = 16×8 = 128 shorts = 8 rows × 16 k-values
            //    Lane i holds rows' k-values at column i? 
            // Actually: a is 8×16 matrix (8 rows, 16 k-elements), distributed as:
            //    Each WI holds short8 = 8 values = one column of the 8×16 tile
            //    WI lane_i holds a[row][lane_i] for row=0..7? No, it's the k-dimension.
            // For DPAS k16: a[m,k] where m=8 (repcount), k=16
            //    Distributed: lane_i holds k=i for all 8 m-rows → short8 per lane
            // So lane_i of short8 a = {a[0,i], a[1,i], ..., a[7,i]} but packed as halfs in short

            // B: int8 = 16×16 matrix (k=16, n=16 per subgroup)
            //    Distributed: lane_i holds column i, int8 = 8 ints = 16 halfs = k=0..15
            // So lane_i of int8 b = {b[0,i]b[1,i], b[2,i]b[3,i], ...} pairs packed as int

            // Load A block 0 (rows 0-7, k 0-15) from SLM
            short8 a0, a1, a2, a3;
            // For a0: rows 0-7, k=0..15. Lane sg_lane reads column sg_lane.
            // a0.s0 = slm[row0 * STRIDE + sg_lane] as short (half)
            #define LOAD_A_ROW(block, row_offset) \
                block.s0 = as_short(cur_slm[(row_offset + 0) * SLM_STRIDE + sg_lane]); \
                block.s1 = as_short(cur_slm[(row_offset + 1) * SLM_STRIDE + sg_lane]); \
                block.s2 = as_short(cur_slm[(row_offset + 2) * SLM_STRIDE + sg_lane]); \
                block.s3 = as_short(cur_slm[(row_offset + 3) * SLM_STRIDE + sg_lane]); \
                block.s4 = as_short(cur_slm[(row_offset + 4) * SLM_STRIDE + sg_lane]); \
                block.s5 = as_short(cur_slm[(row_offset + 5) * SLM_STRIDE + sg_lane]); \
                block.s6 = as_short(cur_slm[(row_offset + 6) * SLM_STRIDE + sg_lane]); \
                block.s7 = as_short(cur_slm[(row_offset + 7) * SLM_STRIDE + sg_lane]);

            LOAD_A_ROW(a0, 0)
            LOAD_A_ROW(a1, 8)
            LOAD_A_ROW(a2, 16)
            LOAD_A_ROW(a3, 24)

            // Load B block (k 0-15, 16 cols for this SG)
            // B[k + k_inner, tile_col + lane] - lane handles column
            // int8 b: lane_i holds 16 halfs (k=0..15) packed as 8 ints
            // Each int = 2 consecutive halfs: b[2j, lane], b[2j+1, lane]
            int8 b0;
            __global const half* b_ptr = B + (k + 0) * N + tile_col + sg_lane;
            // Load 16 rows of B, each row contributes one half per lane
            half b_vals[16];
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                b_vals[i] = b_ptr[i * N];
            }
            // Pack pairs into int
            b0.s0 = as_int((short2)(as_short(b_vals[0]), as_short(b_vals[1])));
            b0.s1 = as_int((short2)(as_short(b_vals[2]), as_short(b_vals[3])));
            b0.s2 = as_int((short2)(as_short(b_vals[4]), as_short(b_vals[5])));
            b0.s3 = as_int((short2)(as_short(b_vals[6]), as_short(b_vals[7])));
            b0.s4 = as_int((short2)(as_short(b_vals[8]), as_short(b_vals[9])));
            b0.s5 = as_int((short2)(as_short(b_vals[10]), as_short(b_vals[11])));
            b0.s6 = as_int((short2)(as_short(b_vals[12]), as_short(b_vals[13])));
            b0.s7 = as_int((short2)(as_short(b_vals[14]), as_short(b_vals[15])));

            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b0, acc0);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b0, acc1);
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b0, acc2);
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b0, acc3);
        }

        // Process k_inner = 16..31 (second k16 block)
        {
            short8 a0, a1, a2, a3;
            // Load from SLM columns 16..31
            #define LOAD_A_ROW2(block, row_offset) \
                block.s0 = as_short(cur_slm[(row_offset + 0) * SLM_STRIDE + 16 + sg_lane]); \
                block.s1 = as_short(cur_slm[(row_offset + 1) * SLM_STRIDE + 16 + sg_lane]); \
                block.s2 = as_short(cur_slm[(row_offset + 2) * SLM_STRIDE + 16 + sg_lane]); \
                block.s3 = as_short(cur_slm[(row_offset + 3) * SLM_STRIDE + 16 + sg_lane]); \
                block.s4 = as_short(cur_slm[(row_offset + 4) * SLM_STRIDE + 16 + sg_lane]); \
                block.s5 = as_short(cur_slm[(row_offset + 5) * SLM_STRIDE + 16 + sg_lane]); \
                block.s6 = as_short(cur_slm[(row_offset + 6) * SLM_STRIDE + 16 + sg_lane]); \
                block.s7 = as_short(cur_slm[(row_offset + 7) * SLM_STRIDE + 16 + sg_lane]);

            LOAD_A_ROW2(a0, 0)
            LOAD_A_ROW2(a1, 8)
            LOAD_A_ROW2(a2, 16)
            LOAD_A_ROW2(a3, 24)

            int8 b1;
            __global const half* b_ptr = B + (k + 16) * N + tile_col + sg_lane;
            half b_vals[16];
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                b_vals[i] = b_ptr[i * N];
            }
            b1.s0 = as_int((short2)(as_short(b_vals[0]), as_short(b_vals[1])));
            b1.s1 = as_int((short2)(as_short(b_vals[2]), as_short(b_vals[3])));
            b1.s2 = as_int((short2)(as_short(b_vals[4]), as_short(b_vals[5])));
            b1.s3 = as_int((short2)(as_short(b_vals[6]), as_short(b_vals[7])));
            b1.s4 = as_int((short2)(as_short(b_vals[8]), as_short(b_vals[9])));
            b1.s5 = as_int((short2)(as_short(b_vals[10]), as_short(b_vals[11])));
            b1.s6 = as_int((short2)(as_short(b_vals[12]), as_short(b_vals[13])));
            b1.s7 = as_int((short2)(as_short(b_vals[14]), as_short(b_vals[15])));

            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b1, acc0);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b1, acc1);
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b1, acc2);
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b1, acc3);
        }

        // Wait for next A tile load to complete
        if (next_k < K) {
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        cur_buf = next_buf;
    }

    // Store results: each subgroup writes 32 rows x 16 cols
    // acc0 = rows 0-7, acc1 = rows 8-15, acc2 = rows 16-23, acc3 = rows 24-31
    // Each float8 has 8 values for 8 rows, lane sg_lane handles column tile_col+sg_lane

    #define STORE_ROW(acc_block, row_base, idx) \
    { \
        int r = tile_row + row_base + idx; \
        int c = tile_col + sg_lane; \
        if (r < M && c < N) \
            C[r * N + c] = convert_half(acc_block[idx]); \
    }

    #define STORE_BLOCK(acc_block, row_base) \
        STORE_ROW(acc_block, row_base, 0) \
        STORE_ROW(acc_block, row_base, 1) \
        STORE_ROW(acc_block, row_base, 2) \
        STORE_ROW(acc_block, row_base, 3) \
        STORE_ROW(acc_block, row_base, 4) \
        STORE_ROW(acc_block, row_base, 5) \
        STORE_ROW(acc_block, row_base, 6) \
        STORE_ROW(acc_block, row_base, 7)

    // Use array access for float8
    float* p0 = (float*)&acc0;
    float* p1 = (float*)&acc1;
    float* p2 = (float*)&acc2;
    float* p3 = (float*)&acc3;

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int r = tile_row + i;
        C[r * N + tile_col + sg_lane] = convert_half(p0[i]);
    }
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int r = tile_row + 8 + i;
        C[r * N + tile_col + sg_lane] = convert_half(p1[i]);
    }
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int r = tile_row + 16 + i;
        C[r * N + tile_col + sg_lane] = convert_half(p2[i]);
    }
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int r = tile_row + 24 + i;
        C[r * N + tile_col + sg_lane] = convert_half(p3[i]);
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
================== 4 passed, 1 deselected, 1 warning in 0.92s ==================
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
1. Memory Alignment: Align data structures to vector boundaries. Use __attribute__((aligned(N))) where N is 16, 32, or 64. Ensure global memory allocations are aligned for coalesced access.
2. Minimize Global Memory Access: Cache frequently accessed values in private variables (registers). Use __constant address space for read-only data to enable hardware caching. Use __global const restrict for read-only buffers to hint compiler optimizations.

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
Branch 0: Syntax precheck failed: clBuildProgram failed: BUILD_PROGRAM_FAILURE - clBuildProgram failed: BUILD_PROGRAM_FAILURE - clBuildProgram failed: BUILD_PROGRAM_FAILURE

Build on <pyopencl.Device 'Intel(R) Arc(TM) B580 Graphics' on 'Intel(R) OpenCL Graphics' at 0x55eb3378ffc0>:

1:98:33: error: invalid reinterpretation: sizes of
[END COMPILE ERROR FEEDBACK]
