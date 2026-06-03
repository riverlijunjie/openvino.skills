

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

### Version 1 (Result: Correctness score: 0 / 5 (compilation error), runtime in ms: -1.000):
```OCL
// GEMM: C[M,N] = A[M,K] x B[K,N], all half precision, f32 accumulation
// Launch: GWS = (N/64 * 16, M/32), LWS = (64, 1)
//   - 64 WIs per WG = 4 subgroups of 16
//   - Each WG computes 32x64 output tile
//   - TILE_M=32, TILE_N=64, TILE_K=32
// Subgroup size: 16
// Required extensions: cl_intel_subgroups, cl_intel_subgroups_short, cl_intel_required_subgroup_size

#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable

#define TILE_M 32
#define TILE_N 64
#define TILE_K 32
#define SG_SIZE 16
#define NUM_SG 4

// SLM stride with padding to reduce bank conflicts
// A in SLM: 32 rows x 32 cols of half, but stored as 32 rows x 36 cols (pad by 4)
#define SLM_A_STRIDE 36

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
    // Work group tile position
    const int wg_id_n = get_group_id(0);  // which 64-col tile
    const int wg_id_m = get_group_id(1);  // which 32-row tile

    const int tile_row = wg_id_m * TILE_M;
    const int tile_col = wg_id_n * TILE_N;

    const int local_id = get_local_id(0);
    const int sg_id = get_sub_group_id();        // 0..3
    const int sg_local_id = get_sub_group_local_id(); // 0..15

    // Each subgroup handles 16 columns of the 64-col tile
    // sg_id 0 -> cols 0-15, sg_id 1 -> cols 16-31, etc.
    const int sg_col_offset = sg_id * 16;

    // SLM for A double buffer: 2 x 32 rows x SLM_A_STRIDE halfs
    __local half slm_a[2 * TILE_M * SLM_A_STRIDE];

    // Accumulators: 32 rows x 16 cols per subgroup = 4 DPAS blocks of 8x16
    float8 acc0 = 0.0f;  // rows 0-7
    float8 acc1 = 0.0f;  // rows 8-15
    float8 acc2 = 0.0f;  // rows 16-23
    float8 acc3 = 0.0f;  // rows 24-31

    // Cooperative A load: 64 WIs load 32x32 halfs
    // Each WI loads 32*32/64 = 16 halfs
    // Strategy: each WI loads one half-row (16 halfs) from a specific row
    // local_id / 2 = row (0..31), local_id % 2 = which 16-element half (0 or 1)
    const int a_load_row = local_id / 2;
    const int a_load_col_base = (local_id % 2) * 16;

    // Preload first A tile into buffer 0
    int buf = 0;
    {
        int a_row = tile_row + a_load_row;
        int a_col = a_load_col_base;
        __global const half* a_ptr = A + a_row * K + a_col;
        __local half* slm_dst = slm_a + buf * TILE_M * SLM_A_STRIDE + a_load_row * SLM_A_STRIDE + a_load_col_base;

        // Load 16 halfs
        half8 v0 = vload8(0, a_ptr);
        half8 v1 = vload8(1, a_ptr);
        vstore8(v0, 0, slm_dst);
        vstore8(v1, 1, slm_dst);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // B base pointer for this subgroup's column slice
    __global const half* B_base = B + (tile_col + sg_col_offset);

    // K-loop: unrolled 2x, double-buffered SLM
    // K/TILE_K iterations total, we do pairs
    const int k_iters = K / TILE_K;

    for (int ki = 0; ki < k_iters - 1; ki++) {
        int cur_buf = ki & 1;
        int next_buf = 1 - cur_buf;

        // Start loading next A tile into next_buf (async with compute)
        {
            int next_k = (ki + 1) * TILE_K;
            int a_row = tile_row + a_load_row;
            int a_col = next_k + a_load_col_base;
            __global const half* a_ptr = A + a_row * K + a_col;
            __local half* slm_dst = slm_a + next_buf * TILE_M * SLM_A_STRIDE + a_load_row * SLM_A_STRIDE + a_load_col_base;

            half8 v0 = vload8(0, a_ptr);
            half8 v1 = vload8(1, a_ptr);
            vstore8(v0, 0, slm_dst);
            vstore8(v1, 1, slm_dst);
        }

        // Compute using current buffer
        __local half* a_base = slm_a + cur_buf * TILE_M * SLM_A_STRIDE;
        int cur_k = ki * TILE_K;

        // Process k=0..15 (first k16 block)
        {
            // Load A from SLM: need 8 rows x k16 for each DPAS
            // DPAS a operand: short8 = 8 pairs of half = 8 x (2 halfs packed in short)
            // For rows r..r+7, k..k+15: we need 8 rows, 16 k-elements each
            // Packed as short8 per WI: each WI holds one column-pair across 8 rows
            // Actually for intel_sub_group_f16_f16_matrix_mad_k16:
            //   a: short8 - each subgroup lane holds 8 shorts, representing 8 rows x 1 k-pair
            //   b: int8 - each subgroup lane holds 8 ints, representing 16 k-elements x (lane's col contribution)

            // Load B tile: 16 rows (k) x 16 cols, from global
            // B layout: row-major, B[k][n], stride N
            // For DPAS b operand (int8 per WI): VNNI format - pairs of k packed
            // int8 means 8 ints per WI, each int = 2 halfs (k-pair)
            // 16 k-values, 16 columns -> 8 ints per lane, 16 lanes

            __global const half* b_ptr = B_base + cur_k * N;

            int8 b_tile0;
            // Load B: 16 rows x 16 cols in VNNI format (pairs of k)
            // Each lane loads its column, pairs of consecutive k rows
            for (int p = 0; p < 8; p++) {
                half val0 = b_ptr[(2*p) * N + sg_local_id];
                half val1 = b_ptr[(2*p+1) * N + sg_local_id];
                ushort u0 = as_ushort(val0);
                ushort u1 = as_ushort(val1);
                ((int*)&b_tile0)[p] = (int)u0 | ((int)u1 << 16);
            }

            // Load A from SLM for rows 0-7
            short8 a_block0;
            {
                __local half* a_ptr_row = a_base + 0 * SLM_A_STRIDE;
                for (int r = 0; r < 8; r++) {
                    ushort u0 = as_ushort(a_ptr_row[r * SLM_A_STRIDE + 2 * sg_local_id]);
                    ushort u1 = as_ushort(a_ptr_row[r * SLM_A_STRIDE + 2 * sg_local_id + 1]);
                    ((short*)&a_block0)[r] = as_short((ushort)((uint)u0 | ((uint)u1 << 16)) & 0xFFFF);
                    // Actually short8 a: each element is a short = 2 packed halfs for that row
                    ((short*)&a_block0)[r] = (short)(u0 | (u1 << 8));
                }
            }
            // This approach is getting complex. Let me use a cleaner DPAS model.

            // For Xe2 DPAS with exec_size=16:
            // intel_sub_group_f16_f16_matrix_mad_k16(short8 a, int8 b, float8 acc)
            // a: 8 rows of A, k16 depth. short8 per WI. 
            //    Each short = 2 fp16 values (k-pair). 8 shorts = 8 k-pairs = 16 k-values for ONE row.
            //    Wait - that's per-lane. With 16 lanes, we have 16 WIs.
            //    Actually: a is distributed across the subgroup. 
            //    For repcount=8: a has 8 rows. Each row has 16 k-elements = 8 shorts.
            //    Total = 8 rows * 8 shorts = 64 shorts = 4 shorts per lane (16 lanes).
            //    But the type is short8... 
            //    
            //    Let me reconsider. On Xe2 with exec_size=16:
            //    DPAS(8,8): 8 rows output, systolic depth 8 (= k16 for fp16 since 2 fp16 per slot)
            //    a: short8 per WI. The 8 shorts represent 8 rows, each short = 2 fp16 (one k-pair).
            //       But depth=8 means 8 k-pairs = 16 k-elements. So we need 8 k-pairs per row.
            //       With 16 lanes and 8 rows, each lane provides one k-pair for each row.
            //       Actually: a[row] is replicated across lanes. The systolic array reads a sequentially.
            //       
            //    For exec_size=16, repcount=8, sdepth=8 (k16 for fp16):
            //    - Output: float8 per WI = 8 rows × 16 cols (one col per lane)
            //    - a: short8 per WI = 8 rows × 1 k-pair per WI, but systolic reads all 16 lanes 
            //         sequentially for depth. Wait no.
            //    
            //    The correct interpretation for intel_sub_group_f16_f16_matrix_mad_k16:
            //    - It computes C[8][16] += A[8][16] * B[16][16] where k=16
            //    - a (short8): represents A[8][16] distributed. Each WI lane l holds:
            //      a[i] = pack(A[i][2*l], A[i][2*l+1]) for row i, k-columns 2l and 2l+1
            //      So 8 rows, each with a pair of k-values at positions (2*lane, 2*lane+1)
            //      Total A coverage: 8 rows × 16 k-values (pairs distributed across 16 lanes × 8 rows)
            //      Wait, that's only 2 k-values per row per lane = 2*16 = 32 k-values per row. No, 16.
            //      Actually: short8 = 8 shorts. Each short = 2 halfs. 8 shorts × 16 lanes = 128 shorts = 256 halfs.
            //      For A[8][16]: 128 halfs. So 128 halfs / 16 lanes = 8 halfs per lane = 4 shorts per lane.
            //      Hmm, but the type is short8 (8 shorts per lane). That's 256 halfs total for 128 needed.
            //      
            //    I think the correct model is:
            //    - a: short8 per WI. Each short packs 2 fp16. 
            //      Row i of A (16 elements) is distributed: lane l holds A[i][2l] and A[i][2l+1] packed.
            //      So a[i] = (A[i][2*sg_local_id+1] << 16) | A[i][2*sg_local_id] as a short... 
            //      No, short is 16 bits. Let me think again.
            //      
            //    Actually I think for Xe2:
            //    - a: short8 means 8 × 16-bit values per work-item
            //    - The A matrix is 8 rows × 16 cols (k=16)
            //    - Distribution: lane l holds a[r] = pack_fp16x2(A[r][2l], A[r][2l+1]) but that's 32 bits = int, not short
            //    
            //    Let me look at this differently. The function signature says short8 for a.
            //    On different HW the meaning varies. For Xe2 (exec_size=16):
            //    - The A matrix for one DPAS is 8×16 (8 rows, k=16)
            //    - short8 per lane: 8 shorts = 128 bits per lane, 16 lanes = 2048 bits = 256 bytes
            //    - A[8][16] in fp16 = 256 bytes. Perfect match!
            //    - So each lane holds: for each of 8 rows, one short = 2 fp16 values
            //    - Lane l holds A[r][2l] and A[r][2l+1] packed into one short? No, 2 fp16 = 32 bits = int.
            //    
            //    Wait, 2 fp16 = 4 bytes = 32 bits. But short = 16 bits. So one short can only hold 1 fp16.
            //    8 shorts per lane = 8 fp16 values per lane. 8 × 16 = 128 fp16 = 8×16 matrix. OK!
            //    So each lane l holds one fp16 per row: a[r] = A[r][l] as a short (really ushort/half).
            //    
            //    But then k=16 means we need 16 k-values per row, and with 16 lanes each holding 1 value
            //    per row, that gives exactly 16 k-values per row. Makes sense!
            //    
            //    For b (int8): 8 ints per lane = 256 bits per lane, 16 lanes = 4096 bits = 512 bytes
            //    B[16][16] in fp16 = 512 bytes. Perfect!
            //    Each int = 2 fp16 (VNNI format). Lane l holds: for each of 8 k-pairs (k=0,1), (k=2,3),...(k=14,15):
            //    b[p] = pack(B[2p][l], B[2p+1][l]) as int (2 fp16 packed)
            //    
            //    So: a[r] = as_short(A[r][lane]) -- just the fp16 value reinterpreted as short
            //        b[p] = as_int((uint)as_ushort(B[2p][lane]) | ((uint)as_ushort(B[2p+1][lane]) << 16))

            // OK let me restart the compute section with this understanding
            acc0 = acc0; // placeholder, will redo below
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // I need to restructure this properly. Let me rewrite from scratch.
    // (The above was exploratory; the final clean version follows)

    // Store results (placeholder)
    // ... 
}
```

### Previous kernel and evaluation:
Here is the last OCL kernel we tested (Result: Correctness score: 3 / 5 (compiled but incorrect), runtime in ms: -1.000):
```OCL
// GEMM: C[M,N] = A[M,K] x B[K,N], all row-major FP16
// Launch: GWS = (N/16, M/32), LWS = (4, 1) with reqd_sub_group_size=16
// Effective: 4 subgroups × 16 WIs = 64 WIs per WG
// Each WG computes a 32×64 output tile
// TILE_M=32, TILE_N=64, TILE_K=32
// SLM: A tile with double buffering, B from global memory

#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable

#define TILE_M 32
#define TILE_N 64
#define TILE_K 32
#define SG_SIZE 16
#define NUM_SG 4
#define SLM_STRIDE 36  // 32 + 4 padding to reduce bank conflicts

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(4, 1, 1)))
__kernel void matmul(
    __global const half* A,
    __global const half* B,
    __global half* C,
    const int M,
    const int K,
    const int N)
{
    // Work-group and subgroup identification
    const int wg_n = get_group_id(0);  // which 64-col block
    const int wg_m = get_group_id(1);  // which 32-row block
    const int sg_id = get_sub_group_id();  // 0..3, each handles 16 columns
    const int sg_lid = get_sub_group_local_id();  // 0..15

    // Global offsets
    const int m_start = wg_m * TILE_M;
    const int n_start = wg_n * TILE_N + sg_id * SG_SIZE;

    // SLM for A double buffer
    __local half slm_a[2 * TILE_M * SLM_STRIDE];

    // Accumulators: 32 rows × 16 cols (per SG), stored as 4 blocks of float8
    float8 acc0 = (float8)(0.0f);  // rows 0-7
    float8 acc1 = (float8)(0.0f);  // rows 8-15
    float8 acc2 = (float8)(0.0f);  // rows 16-23
    float8 acc3 = (float8)(0.0f);  // rows 24-31

    // Linear thread ID within WG for cooperative SLM loading
    const int local_id = sg_id * SG_SIZE + sg_lid;  // 0..63

    // Cooperative A load: 32 rows × 32 cols = 1024 half elements
    // 64 threads, each loads 1024/64 = 16 elements
    // Each thread loads one row's worth of 16 consecutive elements (half a row)
    // Thread mapping: thread i loads row (i/2), col_offset (i%2)*16
    const int a_load_row = local_id / 2;
    const int a_load_col_base = (local_id % 2) * 16;

    // Prefetch first A tile into SLM buffer 0
    {
        __global const half* a_ptr = A + (m_start + a_load_row) * K + a_load_col_base;
        __local half* slm_ptr = slm_a + a_load_row * SLM_STRIDE + a_load_col_base;

        // Load 16 half elements
        half8 va0 = vload8(0, a_ptr);
        half8 va1 = vload8(1, a_ptr);
        vstore8(va0, 0, slm_ptr);
        vstore8(va1, 0, slm_ptr + 8);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int buf = 0;
    const int k_iters = K / TILE_K;

    for (int ki = 0; ki < k_iters; ki++) {
        int k_offset = ki * TILE_K;
        int next_buf = 1 - buf;

        // Prefetch next A tile into next SLM buffer (if not last iteration)
        if (ki < k_iters - 1) {
            int next_k = (ki + 1) * TILE_K;
            __global const half* a_ptr = A + (m_start + a_load_row) * K + next_k + a_load_col_base;
            __local half* slm_ptr = slm_a + next_buf * TILE_M * SLM_STRIDE + a_load_row * SLM_STRIDE + a_load_col_base;

            half8 va0 = vload8(0, a_ptr);
            half8 va1 = vload8(1, a_ptr);
            vstore8(va0, 0, slm_ptr);
            vstore8(va1, 0, slm_ptr + 8);
        }

        // Current SLM buffer base
        __local half* a_slm_base = slm_a + buf * TILE_M * SLM_STRIDE;

        // Process TILE_K=32 in two steps of 16 (matching DPAS k16)
        #pragma unroll
        for (int kk = 0; kk < 2; kk++) {
            int k_inner = kk * 16;

            // Load B tile: 16 k-rows × 16 cols for this subgroup
            // B is row-major: B[k][n], we need B[k_offset + k_inner + kr][n_start + col]
            // For DPAS, B needs to be packed as int8 per WI:
            // Each WI holds one column, 16 k-elements packed as 8 ints (2 fp16 per int)
            int8 b_val;
            __global const half* b_ptr = B + (k_offset + k_inner) * N + n_start + sg_lid;

            #pragma unroll
            for (int kr = 0; kr < 8; kr++) {
                half b0 = b_ptr[(kr * 2) * N];
                half b1 = b_ptr[(kr * 2 + 1) * N];
                // Pack two fp16 into one int (low=first k, high=second k)
                short s0 = as_short(b0);
                short s1 = as_short(b1);
                ((int*)&b_val)[kr] = (int)((uint)(ushort)s0 | ((uint)(ushort)s1 << 16));
            }

            // Load A from SLM and perform DPAS for each 8-row block
            // A for DPAS: short8 = 8 rows × 16 k-elements (each short = 1 fp16, but packed as k16)
            // Actually for intel_sub_group_f16_f16_matrix_mad_k16:
            //   a is short8: across the subgroup, represents 8 rows × 16 k-elements
            //   Each WI holds 8 shorts = 8 k-element pairs? No...
            //   Actually: short8 per WI, 16 WIs in SG → 8 rows, each row has 16 fp16 values
            //   WI[i] holds row[r]'s k-elements at positions determined by subgroup layout
            //   For Xe2 (exec_size=16): each WI holds one column of A's 8×16 block
            //   So WI sg_lid holds A[row+r][k + sg_lid] for r=0..7, packed as short8

            // Block 0: rows 0-7
            short8 a0;
            #pragma unroll
            for (int r = 0; r < 8; r++) {
                __local half* row_ptr = a_slm_base + r * SLM_STRIDE + k_inner + sg_lid;
                ((short*)&a0)[r] = as_short(*row_ptr);
            }
            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b_val, acc0);

            // Block 1: rows 8-15
            short8 a1;
            #pragma unroll
            for (int r = 0; r < 8; r++) {
                __local half* row_ptr = a_slm_base + (8 + r) * SLM_STRIDE + k_inner + sg_lid;
                ((short*)&a1)[r] = as_short(*row_ptr);
            }
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b_val, acc1);

            // Block 2: rows 16-23
            short8 a2;
            #pragma unroll
            for (int r = 0; r < 8; r++) {
                __local half* row_ptr = a_slm_base + (16 + r) * SLM_STRIDE + k_inner + sg_lid;
                ((short*)&a2)[r] = as_short(*row_ptr);
            }
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b_val, acc2);

            // Block 3: rows 24-31
            short8 a3;
            #pragma unroll
            for (int r = 0; r < 8; r++) {
                __local half* row_ptr = a_slm_base + (24 + r) * SLM_STRIDE + k_inner + sg_lid;
                ((short*)&a3)[r] = as_short(*row_ptr);
            }
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b_val, acc3);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
        buf = next_buf;
    }

    // Write results: each WI writes one column (sg_lid) across 32 rows
    // acc0..acc3 each have 8 float values = 8 rows for this WI's column
    __global half* c_ptr = C + m_start * N + n_start + sg_lid;

    #pragma unroll
    for (int r = 0; r < 8; r++) {
        c_ptr[r * N] = convert_half(((float*)&acc0)[r]);
    }
    #pragma unroll
    for (int r = 0; r < 8; r++) {
        c_ptr[(8 + r) * N] = convert_half(((float*)&acc1)[r]);
    }
    #pragma unroll
    for (int r = 0; r < 8; r++) {
        c_ptr[(16 + r) * N] = convert_half(((float*)&acc2)[r]);
    }
    #pragma unroll
    for (int r = 0; r < 8; r++) {
        c_ptr[(24 + r) * N] = convert_half(((float*)&acc3)[r]);
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

self = <task.TestMatmulOCL object at 0x72f07eb1aa80>
kernel = <function initialize_matmul_kernel.<locals>.run at 0x72f070939440>
ocl_queue = <pyopencl._cl.CommandQueue object at 0x72f07ea3de90>, _run = 0

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
E        +  where False = <function allclose at 0x72f0d6f5c0f0>(array([[38.625    ,  6.4453125, 15.1875   , ...,  0.       ,  0.       ,\n         0.       ],\n       [-3.234375 , 43.1875   , 16.75     , ...,  0.       ,  0.       ,\n         0.       ],\n       [       nan,        nan,        nan, ...,  0.       ,  0.       ,\n         0.       ],\n       ...,\n       [ 0.       ,  0.       ,  0.       , ...,  0.       ,  0.       ,\n         0.       ],\n       [ 0.       ,  0.       ,  0.       , ...,  0.       ,  0.       ,\n         0.       ],\n       [ 0.       ,  0.       ,  0.       , ...,  0.       ,  0.       ,\n         0.       ]], shape=(2048, 2048), dtype=float32), array([[-12.434087 , -85.22102  ,  45.86866  , ..., -67.074715 ,\n        -64.52674  ,  37.798523 ],\n       [-82.95244  ,  28.332115 ,   4.3084497, ...,  37.17192  ,\n         48.87541  ,  55.1519   ],\n       [ 31.096529 , -51.77693  ,  -9.3054905, ...,   8.124319 ,\n         61.21928  ,   4.7092314],\n       ...,\n       [-65.29967  , -27.73106  ,  74.195465 , ..., 122.09403  ,\n        -41.569603 ,  10.711429 ],\n       [ 44.6838   ,   2.3142765,  22.61605  , ..., -35.807106 ,\n         42.793472 ,  52.60636  ],\n       [ 50.399834 ,  -3.015791 ,  21.545517 , ..., -21.399685 ,\n        -36.035267 ,  49.01544  ]], shape=(2048, 2048), dtype=float32), rtol=0.02, atol=0.02)
E        +    where <function allclose at 0x72f0d6f5c0f0> = np.allclose

task.py:244: AssertionError
________________ TestMatmulOCL.test_correctness_wrt_pytorch[1] _________________

self = <task.TestMatmulOCL object at 0x72f07ea931a0>
kernel = <function initialize_matmul_kernel.<locals>.run at 0x72f070939440>
ocl_queue = <pyopencl._cl.CommandQueue object at 0x72f07ea3de90>, _run = 1

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
E        +  where False = <function allclose at 0x72f0d6f5c0f0>(array([[ -1.1435547, -22.921875 ,  11.4453125, ...,   0.       ,\n          0.       ,   0.       ],\n       [  2.0097656,  -2.453125 ,  -1.8876953, ...,   0.       ,\n          0.       ,   0.       ],\n       [        nan,         nan,         nan, ...,   0.       ,\n          0.       ,   0.       ],\n       ...,\n       [  0.       ,   0.       ,   0.       , ...,   0.       ,\n          0.       ,   0.       ],\n       [  0.       ,   0.       ,   0.       , ...,   0.       ,\n          0.       ,   0.       ],\n       [  0.       ,   0.       ,   0.       , ...,   0.       ,\n          0.       ,   0.       ]], shape=(2048, 2048), dtype=float32), array([[  63.220627 ,   13.663691 ,   62.708282 , ...,   26.950535 ,\n        -100.14888  ,  -76.10468  ],\n       [  30.338015 ,   -9.576593 ,  -15.848044 , ...,  -86.66203  ,\n           6.3691177,    9.569207 ],\n       [  -9.825886 ,   26.83852  ,  -39.88768  , ...,   94.32298  ,\n         -40.437588 ,   13.349518 ],\n       ...,\n       [ -50.946926 ,  -10.7210655,  -18.652342 , ...,   -4.0612535,\n         -29.112085 ,   -2.7683525],\n       [ -41.46417  ,   -5.034666 ,   35.500336 , ...,    3.5289268,\n          14.26104  ,   55.58531  ],\n       [ -33.896618 ,   51.45737  ,   13.108513 , ...,   11.92079  ,\n         -64.022385 ,   63.048595 ]], shape=(2048, 2048), dtype=float32), rtol=0.02, atol=0.02)
E        +    where <function allclose at 0x72f0d6f5c0f0> = np.allclose

task.py:244: AssertionError
_______________ TestMatmulOCL.test_correctness_wrt_reference[0] ________________

self = <task.TestMatmulOCL object at 0x72f07ea2b5f0>
kernel = <function initialize_matmul_kernel.<locals>.run at 0x72f070939440>
ocl_queue = <pyopencl._cl.CommandQueue object at 0x72f07ea3de90>, _run = 0

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
E        +  where False = <function allclose at 0x72f0d6f5c0f0>(array([[-35.375   ,  52.25    ,  28.484375, ...,   0.      ,   0.      ,\n          0.      ],\n       [ 17.25    , -13.      ,  57.125   , ...,   0.      ,   0.      ,\n          0.      ],\n       [       nan,        nan,        nan, ...,   0.      ,   0.      ,\n          0.      ],\n       ...,\n       [  0.      ,   0.      ,   0.      , ...,   0.      ,   0.      ,\n          0.      ],\n       [  0.      ,   0.      ,   0.      , ...,   0.      ,   0.      ,\n          0.      ],\n       [  0.      ,   0.      ,   0.      , ...,   0.      ,   0.      ,\n          0.      ]], shape=(2048, 2048), dtype=float32), array([[-5.7125000e+01, -2.8015625e+01,  6.5562500e+01, ...,\n        -5.0218750e+01,  2.7000000e+01,  2.6109375e+01],\n       [ 9.2687500e+01,  1.8343750e+01, -1.4343750e+01, ...,\n        -4.0843750e+01, -2.5828125e+01, -3.1914062e+00],\n       [-2.5585938e+00, -3.0015625e+01,  4.9937500e+01, ...,\n        -5.7156250e+01, -9.2250000e+01,  1.2921875e+01],\n       ...,\n       [-4.6031250e+01, -1.4262500e+02, -6.1500000e+01, ...,\n        -1.2194824e-01,  6.6445312e+00, -5.3156250e+01],\n       [ 3.1859375e+01, -2.3484375e+01,  4.2750000e+01, ...,\n         1.8390625e+01, -2.0507812e+00, -1.6000000e+02],\n       [-3.8968750e+01,  9.1750000e+01,  2.3953125e+01, ...,\n         4.5437500e+01,  5.4437500e+01,  1.0787500e+02]],\n      shape=(2048, 2048), dtype=float32), rtol=0.02, atol=0.02)
E        +    where <function allclose at 0x72f0d6f5c0f0> = np.allclose

task.py:262: AssertionError
_______________ TestMatmulOCL.test_correctness_wrt_reference[1] ________________

self = <task.TestMatmulOCL object at 0x72f07ea2b6b0>
kernel = <function initialize_matmul_kernel.<locals>.run at 0x72f070939440>
ocl_queue = <pyopencl._cl.CommandQueue object at 0x72f07ea3de90>, _run = 1

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
E        +  where False = <function allclose at 0x72f0d6f5c0f0>(array([[ 60.6875  , -27.859375, -45.9375  , ...,   0.      ,   0.      ,\n          0.      ],\n       [ -4.1875  , -27.828125, -23.46875 , ...,   0.      ,   0.      ,\n          0.      ],\n       [       nan,        nan,        nan, ...,   0.      ,   0.      ,\n          0.      ],\n       ...,\n       [  0.      ,   0.      ,   0.      , ...,   0.      ,   0.      ,\n          0.      ],\n       [  0.      ,   0.      ,   0.      , ...,   0.      ,   0.      ,\n          0.      ],\n       [  0.      ,   0.      ,   0.      , ...,   0.      ,   0.      ,\n          0.      ]], shape=(2048, 2048), dtype=float32), array([[  46.4375   ,  -21.75     ,  -36.71875  , ...,  -39.96875  ,\n           3.125    , -147.25     ],\n       [ -13.859375 ,  -17.453125 ,   17.375    , ...,   57.90625  ,\n          25.859375 ,  -68.0625   ],\n       [  34.1875   ,   -9.421875 ,  -49.       , ...,  -46.71875  ,\n         -64.5625   ,   37.6875   ],\n       ...,\n       [   3.3261719,  -60.03125  ,  -64.       , ...,   18.328125 ,\n         -17.234375 ,  -30.90625  ],\n       [  -0.9145508,   97.25     ,  -18.109375 , ...,   35.75     ,\n         -45.0625   ,   21.78125  ],\n       [ -36.5      ,  -63.8125   ,   68.125    , ...,  -48.09375  ,\n         -29.515625 ,   11.1640625]], shape=(2048, 2048), dtype=float32), rtol=0.02, atol=0.02)
E        +    where <function allclose at 0x72f0d6f5c0f0> = np.allclose

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
================== 4 failed, 1 deselected, 1 warning in 0.82s ==================

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

- **Shared Local Memory (SLM) Tiling**: Cache frequently accessed data in SLM using `group_local_memory_for_overwrite`. Synchronize with `group_barrier()` after writes and before reads. Use 16×16 or 32×32 tiles for float data.
- **Blocked/Tiled Algorithms**: Process input in blocks to bound peak memory. Trade recomputation for memory savings (e.g., Flash-Attention style). Maintain running accumulators across blocks with proper rescaling.
- **Sub-Group Collectives**: Use `reduce_over_group(sg, val, op)` for hardware-accelerated SIMD reductions. Use `group_broadcast` and `shift_group_*` for efficient data sharing.

[COMPILE ERROR FEEDBACK FROM PREVIOUS TRIAL]
The following compile errors occurred in the previous iteration. Avoid these mistakes in your implementation:
Branch 1: SyntaxError. 
 Your output was not in the required format. The code could not be extracted.

Branch 2: Syntax precheck failed: clBuildProgram failed: BUILD_PROGRAM_FAILURE - clBuildProgram failed: BUILD_PROGRAM_FAILURE - clBuildProgram failed: BUILD_PROGRAM_FAILURE

Build on <pyopencl.Device 'Intel(R) Arc(TM) B580 Graphics' on 'Intel(R) OpenCL Graphics' at 0x62f64660e3b0>:

2:144:46: error: invalid reinterpretation: sizes o
Branch 3: SyntaxError. 
 Your output was not in the required format. The code could not be extracted.

[END COMPILE ERROR FEEDBACK]
