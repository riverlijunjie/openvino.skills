

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

### Version 1 (Result: Correctness score: 5 / 5 (compiles and correct), runtime in ms: 1.510):
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

## Best implementation so far:

This is the code of the best kernel we have tested so far (Result: Correctness score: 5 / 5 (compiles and correct), runtime in ms: 1.410):
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
    const int wg_n = get_group_id(0);
    const int wg_m = get_group_id(1);
    const int sg_id = get_local_id(1);
    const int sg_lane = get_local_id(0);

    const int tile_row = wg_m * 32;
    const int tile_col = wg_n * 64 + sg_id * 16;

    // Linear local ID for cooperative loads
    const int lid = sg_id * 16 + sg_lane; // 0..63

    // SLM for A: double-buffered, 32 rows x 32 cols of half
    // Stride = 32 halfs (no padding - 16 lanes access consecutive halfs, no bank conflict)
    #define SLM_STRIDE 32
    #define SLM_BUF_SIZE (32 * SLM_STRIDE)
    __local half slm_a[2 * SLM_BUF_SIZE];

    // Accumulators: 32 rows x 16 cols per subgroup = 4 blocks of 8 rows
    float8 acc0 = 0.0f;
    float8 acc1 = 0.0f;
    float8 acc2 = 0.0f;
    float8 acc3 = 0.0f;

    // Cooperative A load helper: 64 WIs load 32x32 = 1024 halfs = 16 per WI
    // lid/2 = row (0..31), (lid%2)*16 = col offset (0 or 16)
    const int load_row = lid / 2;
    const int load_col = (lid % 2) * 16;
    const int slm_store_offset = load_row * SLM_STRIDE + load_col;

    // Preload first A tile into SLM buffer 0
    {
        __global const half* a_ptr = A + (tile_row + load_row) * K + load_col;
        vstore16(vload16(0, a_ptr), 0, &slm_a[slm_store_offset]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int cur_buf = 0;

    // Main K-loop, step by 32. K is guaranteed divisible by 32.
    for (int k = 0; k < K; k += 32) {
        int next_k = k + 32;
        int next_buf = 1 - cur_buf;

        __local const half* cur_slm = &slm_a[cur_buf * SLM_BUF_SIZE];

        // ===== K-step 0: k_inner = 0..15 =====
        // Load A from SLM for rows 0-31, k cols 0-15
        short8 a0, a1, a2, a3;

        a0 = (short8)(
            as_short(cur_slm[0 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[1 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[2 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[3 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[4 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[5 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[6 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[7 * SLM_STRIDE + sg_lane])
        );
        a1 = (short8)(
            as_short(cur_slm[8 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[9 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[10 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[11 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[12 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[13 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[14 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[15 * SLM_STRIDE + sg_lane])
        );
        a2 = (short8)(
            as_short(cur_slm[16 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[17 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[18 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[19 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[20 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[21 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[22 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[23 * SLM_STRIDE + sg_lane])
        );
        a3 = (short8)(
            as_short(cur_slm[24 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[25 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[26 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[27 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[28 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[29 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[30 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[31 * SLM_STRIDE + sg_lane])
        );

        // Load B for k_inner=0..15: B[k..k+15, tile_col+sg_lane]
        // Each lane loads 16 halfs from 16 consecutive k-rows at its column
        // Pack pairs into int for VNNI format
        int8 b0;
        {
            __global const half* b_ptr = B + k * N + tile_col + sg_lane;
            short bv0  = as_short(b_ptr[0 * N]);
            short bv1  = as_short(b_ptr[1 * N]);
            short bv2  = as_short(b_ptr[2 * N]);
            short bv3  = as_short(b_ptr[3 * N]);
            short bv4  = as_short(b_ptr[4 * N]);
            short bv5  = as_short(b_ptr[5 * N]);
            short bv6  = as_short(b_ptr[6 * N]);
            short bv7  = as_short(b_ptr[7 * N]);
            short bv8  = as_short(b_ptr[8 * N]);
            short bv9  = as_short(b_ptr[9 * N]);
            short bv10 = as_short(b_ptr[10 * N]);
            short bv11 = as_short(b_ptr[11 * N]);
            short bv12 = as_short(b_ptr[12 * N]);
            short bv13 = as_short(b_ptr[13 * N]);
            short bv14 = as_short(b_ptr[14 * N]);
            short bv15 = as_short(b_ptr[15 * N]);

            b0.s0 = as_int((short2)(bv0, bv1));
            b0.s1 = as_int((short2)(bv2, bv3));
            b0.s2 = as_int((short2)(bv4, bv5));
            b0.s3 = as_int((short2)(bv6, bv7));
            b0.s4 = as_int((short2)(bv8, bv9));
            b0.s5 = as_int((short2)(bv10, bv11));
            b0.s6 = as_int((short2)(bv12, bv13));
            b0.s7 = as_int((short2)(bv14, bv15));
        }

        // DPAS for k_inner=0..15
        acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b0, acc0);
        acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b0, acc1);
        acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b0, acc2);
        acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b0, acc3);

        // ===== K-step 1: k_inner = 16..31 =====
        // Load A from SLM for k cols 16-31
        short8 a4, a5, a6, a7;

        a4 = (short8)(
            as_short(cur_slm[0 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[1 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[2 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[3 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[4 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[5 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[6 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[7 * SLM_STRIDE + 16 + sg_lane])
        );
        a5 = (short8)(
            as_short(cur_slm[8 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[9 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[10 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[11 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[12 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[13 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[14 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[15 * SLM_STRIDE + 16 + sg_lane])
        );
        a6 = (short8)(
            as_short(cur_slm[16 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[17 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[18 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[19 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[20 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[21 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[22 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[23 * SLM_STRIDE + 16 + sg_lane])
        );
        a7 = (short8)(
            as_short(cur_slm[24 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[25 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[26 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[27 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[28 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[29 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[30 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[31 * SLM_STRIDE + 16 + sg_lane])
        );

        // Load B for k_inner=16..31
        int8 b1;
        {
            __global const half* b_ptr = B + (k + 16) * N + tile_col + sg_lane;
            short bv0  = as_short(b_ptr[0 * N]);
            short bv1  = as_short(b_ptr[1 * N]);
            short bv2  = as_short(b_ptr[2 * N]);
            short bv3  = as_short(b_ptr[3 * N]);
            short bv4  = as_short(b_ptr[4 * N]);
            short bv5  = as_short(b_ptr[5 * N]);
            short bv6  = as_short(b_ptr[6 * N]);
            short bv7  = as_short(b_ptr[7 * N]);
            short bv8  = as_short(b_ptr[8 * N]);
            short bv9  = as_short(b_ptr[9 * N]);
            short bv10 = as_short(b_ptr[10 * N]);
            short bv11 = as_short(b_ptr[11 * N]);
            short bv12 = as_short(b_ptr[12 * N]);
            short bv13 = as_short(b_ptr[13 * N]);
            short bv14 = as_short(b_ptr[14 * N]);
            short bv15 = as_short(b_ptr[15 * N]);

            b1.s0 = as_int((short2)(bv0, bv1));
            b1.s1 = as_int((short2)(bv2, bv3));
            b1.s2 = as_int((short2)(bv4, bv5));
            b1.s3 = as_int((short2)(bv6, bv7));
            b1.s4 = as_int((short2)(bv8, bv9));
            b1.s5 = as_int((short2)(bv10, bv11));
            b1.s6 = as_int((short2)(bv12, bv13));
            b1.s7 = as_int((short2)(bv14, bv15));
        }

        // Start loading next A tile into next buffer (overlaps with DPAS below)
        if (next_k < K) {
            __global const half* a_ptr = A + (tile_row + load_row) * K + next_k + load_col;
            vstore16(vload16(0, a_ptr), 0, &slm_a[next_buf * SLM_BUF_SIZE + slm_store_offset]);
        }

        // DPAS for k_inner=16..31
        acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a4, b1, acc0);
        acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a5, b1, acc1);
        acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a6, b1, acc2);
        acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a7, b1, acc3);

        // Barrier to ensure next A tile is fully loaded before we use it
        barrier(CLK_LOCAL_MEM_FENCE);
        cur_buf = next_buf;
    }

    // Store results: each subgroup writes 32 rows x 16 cols
    // Use intel_sub_group_block_write_us for vectorized stores
    __global half* c_ptr = C + tile_row * N + tile_col + sg_lane;

    float* p0 = (float*)&acc0;
    float* p1 = (float*)&acc1;
    float* p2 = (float*)&acc2;
    float* p3 = (float*)&acc3;

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        c_ptr[(0 + i) * N] = convert_half(p0[i]);
    }
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        c_ptr[(8 + i) * N] = convert_half(p1[i]);
    }
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        c_ptr[(16 + i) * N] = convert_half(p2[i]);
    }
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        c_ptr[(24 + i) * N] = convert_half(p3[i]);
    }
}
```

### Previous kernel and evaluation:
Here is the last OCL kernel we tested (Result: Correctness score: 5 / 5 (compiles and correct), runtime in ms: 1.590):
```OCL
// GEMM: C[M,N] = A[M,K] x B[K,N], all half, f32 accumulation
// Launch: GWS = ((N/64)*16, (M/32)*4, 1), LWS = (16, 4, 1)
// 4 subgroups of 16 WIs per WG. Each WG computes 32x64 tile.
// Each subgroup computes 32x16 using DPAS.
// A in SLM (double-buffered), B from global/L2.
// TILE_M=32, TILE_N=64, TILE_K=32. K must be divisible by 32.
// K-loop unrolled 2x (processes 64 K per iteration).

#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(16, 4, 1)))
__kernel void matmul(
    __global const half* restrict A,
    __global const half* restrict B,
    __global half* restrict C,
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
    // Stride = 34 halfs to avoid bank conflicts
    #define SLM_STRIDE 34
    #define SLM_BUF_SIZE (32 * SLM_STRIDE)
    __local half slm_a[2 * SLM_BUF_SIZE];

    // Accumulators: 32 rows x 16 cols per subgroup = 4 blocks of 8 rows
    float8 acc0 = 0.0f;
    float8 acc1 = 0.0f;
    float8 acc2 = 0.0f;
    float8 acc3 = 0.0f;

    // Cooperative A load helper indices (constant across loop)
    // 64 WIs load 32x32 = 1024 halfs, 16 halfs each
    // 2 chunks per row (32/16=2), so lid/2 = row, lid%2 = chunk
    const int chunk_row = lid / 2;
    const int chunk_col = (lid & 1) * 16;
    const int slm_store_offset_buf0 = chunk_row * SLM_STRIDE + chunk_col;
    const int slm_store_offset_buf1 = SLM_BUF_SIZE + chunk_row * SLM_STRIDE + chunk_col;
    const int a_row_global = tile_row + chunk_row;

    // Preload first A tile (k=0) into SLM buffer 0
    {
        __global const half* a_ptr = A + a_row_global * K + chunk_col;
        half16 a_val = vload16(0, a_ptr);
        vstore16(a_val, 0, &slm_a[slm_store_offset_buf0]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // B column base for this subgroup (constant)
    const int b_col = tile_col + sg_lane;

    // Main K-loop: unrolled 2x, processes 64 K-elements per iteration
    // K is guaranteed divisible by 32 (K=2048), so K/32 = 64 iterations
    // We unroll 2x: 32 iterations of 64 K-elements each
    int cur_buf = 0;

    for (int k = 0; k < K; k += 64) {
        // ============ PHASE 1: Compute k..k+31 from buf cur_buf ============
        // While computing, load A for k+32..k+63 into other buffer
        int next_buf = 1 - cur_buf;

        // Start loading next A tile (k+32) into next_buf
        {
            int a_col = k + 32 + chunk_col;
            __global const half* a_ptr = A + a_row_global * K + a_col;
            half16 a_val = vload16(0, a_ptr);
            int slm_off = next_buf * SLM_BUF_SIZE + chunk_row * SLM_STRIDE + chunk_col;
            vstore16(a_val, 0, &slm_a[slm_off]);
        }

        // Compute from current buffer
        __local half* cur_slm = &slm_a[cur_buf * SLM_BUF_SIZE];

        // --- K-step 0: k_inner = 0..15 ---
        short8 a0_0, a1_0, a2_0, a3_0;
        // Load A from SLM: lane sg_lane reads k-column sg_lane for each row
        a0_0 = (short8)(
            as_short(cur_slm[0 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[1 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[2 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[3 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[4 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[5 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[6 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[7 * SLM_STRIDE + sg_lane]));
        a1_0 = (short8)(
            as_short(cur_slm[8 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[9 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[10 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[11 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[12 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[13 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[14 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[15 * SLM_STRIDE + sg_lane]));
        a2_0 = (short8)(
            as_short(cur_slm[16 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[17 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[18 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[19 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[20 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[21 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[22 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[23 * SLM_STRIDE + sg_lane]));
        a3_0 = (short8)(
            as_short(cur_slm[24 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[25 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[26 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[27 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[28 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[29 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[30 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[31 * SLM_STRIDE + sg_lane]));

        // Load B for k_inner=0..15: B[k..k+15, b_col]
        // VNNI format: pairs of consecutive k-rows packed into int
        int8 b0;
        {
            __global const half* b_ptr = B + k * N + b_col;
            // Load pairs of halfs from consecutive rows and pack as int
            half2 p0 = (half2)(b_ptr[0 * N], b_ptr[1 * N]);
            half2 p1 = (half2)(b_ptr[2 * N], b_ptr[3 * N]);
            half2 p2 = (half2)(b_ptr[4 * N], b_ptr[5 * N]);
            half2 p3 = (half2)(b_ptr[6 * N], b_ptr[7 * N]);
            half2 p4 = (half2)(b_ptr[8 * N], b_ptr[9 * N]);
            half2 p5 = (half2)(b_ptr[10 * N], b_ptr[11 * N]);
            half2 p6 = (half2)(b_ptr[12 * N], b_ptr[13 * N]);
            half2 p7 = (half2)(b_ptr[14 * N], b_ptr[15 * N]);
            b0 = (int8)(as_int(p0), as_int(p1), as_int(p2), as_int(p3),
                        as_int(p4), as_int(p5), as_int(p6), as_int(p7));
        }

        // DPAS for k=0..15
        acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0_0, b0, acc0);
        acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1_0, b0, acc1);
        acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2_0, b0, acc2);
        acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3_0, b0, acc3);

        // --- K-step 1: k_inner = 16..31 ---
        short8 a0_1, a1_1, a2_1, a3_1;
        a0_1 = (short8)(
            as_short(cur_slm[0 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[1 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[2 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[3 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[4 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[5 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[6 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[7 * SLM_STRIDE + 16 + sg_lane]));
        a1_1 = (short8)(
            as_short(cur_slm[8 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[9 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[10 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[11 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[12 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[13 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[14 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[15 * SLM_STRIDE + 16 + sg_lane]));
        a2_1 = (short8)(
            as_short(cur_slm[16 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[17 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[18 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[19 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[20 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[21 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[22 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[23 * SLM_STRIDE + 16 + sg_lane]));
        a3_1 = (short8)(
            as_short(cur_slm[24 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[25 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[26 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[27 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[28 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[29 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[30 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[31 * SLM_STRIDE + 16 + sg_lane]));

        // Load B for k_inner=16..31
        int8 b1;
        {
            __global const half* b_ptr = B + (k + 16) * N + b_col;
            half2 p0 = (half2)(b_ptr[0 * N], b_ptr[1 * N]);
            half2 p1 = (half2)(b_ptr[2 * N], b_ptr[3 * N]);
            half2 p2 = (half2)(b_ptr[4 * N], b_ptr[5 * N]);
            half2 p3 = (half2)(b_ptr[6 * N], b_ptr[7 * N]);
            half2 p4 = (half2)(b_ptr[8 * N], b_ptr[9 * N]);
            half2 p5 = (half2)(b_ptr[10 * N], b_ptr[11 * N]);
            half2 p6 = (half2)(b_ptr[12 * N], b_ptr[13 * N]);
            half2 p7 = (half2)(b_ptr[14 * N], b_ptr[15 * N]);
            b1 = (int8)(as_int(p0), as_int(p1), as_int(p2), as_int(p3),
                        as_int(p4), as_int(p5), as_int(p6), as_int(p7));
        }

        // DPAS for k=16..31
        acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0_1, b1, acc0);
        acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1_1, b1, acc1);
        acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2_1, b1, acc2);
        acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3_1, b1, acc3);

        // Wait for next A tile (k+32) to be ready in SLM
        barrier(CLK_LOCAL_MEM_FENCE);

        // ============ PHASE 2: Compute k+32..k+63 from buf next_buf ============
        // Load A for k+64 into cur_buf (which is now free)
        if (k + 64 < K) {
            int a_col = k + 64 + chunk_col;
            __global const half* a_ptr = A + a_row_global * K + a_col;
            half16 a_val = vload16(0, a_ptr);
            int slm_off = cur_buf * SLM_BUF_SIZE + chunk_row * SLM_STRIDE + chunk_col;
            vstore16(a_val, 0, &slm_a[slm_off]);
        }

        // Compute from next_buf
        __local half* next_slm = &slm_a[next_buf * SLM_BUF_SIZE];

        // --- K-step 2: k+32, k_inner = 0..15 ---
        short8 a0_2, a1_2, a2_2, a3_2;
        a0_2 = (short8)(
            as_short(next_slm[0 * SLM_STRIDE + sg_lane]),
            as_short(next_slm[1 * SLM_STRIDE + sg_lane]),
            as_short(next_slm[2 * SLM_STRIDE + sg_lane]),
            as_short(next_slm[3 * SLM_STRIDE + sg_lane]),
            as_short(next_slm[4 * SLM_STRIDE + sg_lane]),
            as_short(next_slm[5 * SLM_STRIDE + sg_lane]),
            as_short(next_slm[6 * SLM_STRIDE + sg_lane]),
            as_short(next_slm[7 * SLM_STRIDE + sg_lane]));
        a1_2 = (short8)(
            as_short(next_slm[8 * SLM_STRIDE + sg_lane]),
            as_short(next_slm[9 * SLM_STRIDE + sg_lane]),
            as_short(next_slm[10 * SLM_STRIDE + sg_lane]),
            as_short(next_slm[11 * SLM_STRIDE + sg_lane]),
            as_short(next_slm[12 * SLM_STRIDE + sg_lane]),
            as_short(next_slm[13 * SLM_STRIDE + sg_lane]),
            as_short(next_slm[14 * SLM_STRIDE + sg_lane]),
            as_short(next_slm[15 * SLM_STRIDE + sg_lane]));
        a2_2 = (short8)(
            as_short(next_slm[16 * SLM_STRIDE + sg_lane]),
            as_short(next_slm[17 * SLM_STRIDE + sg_lane]),
            as_short(next_slm[18 * SLM_STRIDE + sg_lane]),
            as_short(next_slm[19 * SLM_STRIDE + sg_lane]),
            as_short(next_slm[20 * SLM_STRIDE + sg_lane]),
            as_short(next_slm[21 * SLM_STRIDE + sg_lane]),
            as_short(next_slm[22 * SLM_STRIDE + sg_lane]),
            as_short(next_slm[23 * SLM_STRIDE + sg_lane]));
        a3_2 = (short8)(
            as_short(next_slm[24 * SLM_STRIDE + sg_lane]),
            as_short(next_slm[25 * SLM_STRIDE + sg_lane]),
            as_short(next_slm[26 * SLM_STRIDE + sg_lane]),
            as_short(next_slm[27 * SLM_STRIDE + sg_lane]),
            as_short(next_slm[28 * SLM_STRIDE + sg_lane]),
            as_short(next_slm[29 * SLM_STRIDE + sg_lane]),
            as_short(next_slm[30 * SLM_STRIDE + sg_lane]),
            as_short(next_slm[31 * SLM_STRIDE + sg_lane]));

        // Load B for k+32, k_inner=0..15
        int8 b2;
        {
            __global const half* b_ptr = B + (k + 32) * N + b_col;
            half2 p0 = (half2)(b_ptr[0 * N], b_ptr[1 * N]);
            half2 p1 = (half2)(b_ptr[2 * N], b_ptr[3 * N]);
            half2 p2 = (half2)(b_ptr[4 * N], b_ptr[5 * N]);
            half2 p3 = (half2)(b_ptr[6 * N], b_ptr[7 * N]);
            half2 p4 = (half2)(b_ptr[8 * N], b_ptr[9 * N]);
            half2 p5 = (half2)(b_ptr[10 * N], b_ptr[11 * N]);
            half2 p6 = (half2)(b_ptr[12 * N], b_ptr[13 * N]);
            half2 p7 = (half2)(b_ptr[14 * N], b_ptr[15 * N]);
            b2 = (int8)(as_int(p0), as_int(p1), as_int(p2), as_int(p3),
                        as_int(p4), as_int(p5), as_int(p6), as_int(p7));
        }

        // DPAS for k+32, k_inner=0..15
        acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0_2, b2, acc0);
        acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1_2, b2, acc1);
        acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2_2, b2, acc2);
        acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3_2, b2, acc3);

        // --- K-step 3: k+32, k_inner = 16..31 ---
        short8 a0_3, a1_3, a2_3, a3_3;
        a0_3 = (short8)(
            as_short(next_slm[0 * SLM_STRIDE + 16 + sg_lane]),
            as_short(next_slm[1 * SLM_STRIDE + 16 + sg_lane]),
            as_short(next_slm[2 * SLM_STRIDE + 16 + sg_lane]),
            as_short(next_slm[3 * SLM_STRIDE + 16 + sg_lane]),
            as_short(next_slm[4 * SLM_STRIDE + 16 + sg_lane]),
            as_short(next_slm[5 * SLM_STRIDE + 16 + sg_lane]),
            as_short(next_slm[6 * SLM_STRIDE + 16 + sg_lane]),
            as_short(next_slm[7 * SLM_STRIDE + 16 + sg_lane]));
        a1_3 = (short8)(
            as_short(next_slm[8 * SLM_STRIDE + 16 + sg_lane]),
            as_short(next_slm[9 * SLM_STRIDE + 16 + sg_lane]),
            as_short(next_slm[10 * SLM_STRIDE + 16 + sg_lane]),
            as_short(next_slm[11 * SLM_STRIDE + 16 + sg_lane]),
            as_short(next_slm[12 * SLM_STRIDE + 16 + sg_lane]),
            as_short(next_slm[13 * SLM_STRIDE + 16 + sg_lane]),
            as_short(next_slm[14 * SLM_STRIDE + 16 + sg_lane]),
            as_short(next_slm[15 * SLM_STRIDE + 16 + sg_lane]));
        a2_3 = (short8)(
            as_short(next_slm[16 * SLM_STRIDE + 16 + sg_lane]),
            as_short(next_slm[17 * SLM_STRIDE + 16 + sg_lane]),
            as_short(next_slm[18 * SLM_STRIDE + 16 + sg_lane]),
            as_short(next_slm[19 * SLM_STRIDE + 16 + sg_lane]),
            as_short(next_slm[20 * SLM_STRIDE + 16 + sg_lane]),
            as_short(next_slm[21 * SLM_STRIDE + 16 + sg_lane]),
            as_short(next_slm[22 * SLM_STRIDE + 16 + sg_lane]),
            as_short(next_slm[23 * SLM_STRIDE + 16 + sg_lane]));
        a3_3 = (short8)(
            as_short(next_slm[24 * SLM_STRIDE + 16 + sg_lane]),
            as_short(next_slm[25 * SLM_STRIDE + 16 + sg_lane]),
            as_short(next_slm[26 * SLM_STRIDE + 16 + sg_lane]),
            as_short(next_slm[27 * SLM_STRIDE + 16 + sg_lane]),
            as_short(next_slm[28 * SLM_STRIDE + 16 + sg_lane]),
            as_short(next_slm[29 * SLM_STRIDE + 16 + sg_lane]),
            as_short(next_slm[30 * SLM_STRIDE + 16 + sg_lane]),
            as_short(next_slm[31 * SLM_STRIDE + 16 + sg_lane]));

        // Load B for k+48, k_inner=0..15
        int8 b3;
        {
            __global const half* b_ptr = B + (k + 48) * N + b_col;
            half2 p0 = (half2)(b_ptr[0 * N], b_ptr[1 * N]);
            half2 p1 = (half2)(b_ptr[2 * N], b_ptr[3 * N]);
            half2 p2 = (half2)(b_ptr[4 * N], b_ptr[5 * N]);
            half2 p3 = (half2)(b_ptr[6 * N], b_ptr[7 * N]);
            half2 p4 = (half2)(b_ptr[8 * N], b_ptr[9 * N]);
            half2 p5 = (half2)(b_ptr[10 * N], b_ptr[11 * N]);
            half2 p6 = (half2)(b_ptr[12 * N], b_ptr[13 * N]);
            half2 p7 = (half2)(b_ptr[14 * N], b_ptr[15 * N]);
            b3 = (int8)(as_int(p0), as_int(p1), as_int(p2), as_int(p3),
                        as_int(p4), as_int(p5), as_int(p6), as_int(p7));
        }

        // DPAS for k+32, k_inner=16..31
        acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0_3, b3, acc0);
        acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1_3, b3, acc1);
        acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2_3, b3, acc2);
        acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3_3, b3, acc3);

        // Wait for next A tile (k+64) load to complete
        barrier(CLK_LOCAL_MEM_FENCE);

        // After this iteration, cur_buf has the data for k+64
        // next_buf was used for k+32 (now consumed)
        // cur_buf now holds k+64 data
        // So for next iteration, cur_buf stays the same (it has the fresh data)
        // Actually: we loaded k+64 into cur_buf, so next iteration should use cur_buf
        // cur_buf remains unchanged (it's the buffer we just loaded into)
    }

    // Store results: each subgroup writes 32 rows x 16 cols
    // acc0=rows 0-7, acc1=rows 8-15, acc2=rows 16-23, acc3=rows 24-31
    // Each float8 element = one row, lane sg_lane = column offset
    __global half* c_ptr = C + tile_row * N + tile_col + sg_lane;

    float* p0 = (float*)&acc0;
    float* p1 = (float*)&acc1;
    float* p2 = (float*)&acc2;
    float* p3 = (float*)&acc3;

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        c_ptr[(i) * N] = convert_half(p0[i]);
    }
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        c_ptr[(8 + i) * N] = convert_half(p1[i]);
    }
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        c_ptr[(16 + i) * N] = convert_half(p2[i]);
    }
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        c_ptr[(24 + i) * N] = convert_half(p3[i]);
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
================== 4 passed, 1 deselected, 1 warning in 0.87s ==================
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
1. Minimize Synchronization: Reduce barrier() calls. Use barrier(CLK_LOCAL_MEM_FENCE) for local memory only, barrier(CLK_GLOBAL_MEM_FENCE) when needed for global. Sub-group operations are barrier-free. Use work-group-level reductions only when necessary.
2. Maximize Occupancy: Balance work-group size with register/local memory usage. Query CL_KERNEL_WORK_GROUP_SIZE and CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE. Typical work-group sizes: 64-256 for compute-bound, 32-128 for memory-bound kernels. Use multiples of wavefront/warp size (32 for NVIDIA, 64 for AMD).

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

[COMPILE ERROR FEEDBACK FROM PREVIOUS TRIAL]
The following compile errors occurred in the previous iteration. Avoid these mistakes in your implementation:
Branch 2: SyntaxError. 
 Your output was not in the required format. The code could not be extracted.

Branch 3: SyntaxError. 
 Your output was not in the required format. The code could not be extracted.

[END COMPILE ERROR FEEDBACK]
