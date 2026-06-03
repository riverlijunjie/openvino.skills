

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
// GEMM: C[M,N] = A[M,K] x B[K,N], all half precision, float accumulation
// Architecture: 64 WIs (4 SGs of 16), TILE 32x64x32, A in SLM, B from global/L2
// Launch: GWS = (N/64 * 16, M/32), LWS = (16, 4) → 64 WIs per WG
// Subgroup size: 16
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void matmul(
    __global const half* A,
    __global const half* B,
    __global half* C,
    const int M,
    const int K,
    const int N)
{
    // Work-group tile: 32 rows (M) x 64 cols (N)
    // Each subgroup: 32 rows x 16 cols
    const int wg_n = get_group_id(0);  // which 64-col tile
    const int wg_m = get_group_id(1);  // which 32-row tile

    const int sg_id = get_sub_group_id();       // 0..3
    const int sg_lid = get_sub_group_local_id(); // 0..15

    const int tile_row = wg_m * 32;
    const int tile_col = wg_n * 64 + sg_id * 16;

    // SLM for A: 32 rows x 32 cols (double buffered)
    // Layout: row-major with stride 34 (32 + 2 padding to avoid bank conflicts)
    // Total SLM: 2 * 32 * 34 * 2 bytes = 4352 bytes
    #define SLM_STRIDE 34
    __local half slm_A[2 * 32 * SLM_STRIDE];

    // Accumulators: 4 DPAS results of float8 each = 32 rows x 16 cols
    float8 acc0 = 0.0f;
    float8 acc1 = 0.0f;
    float8 acc2 = 0.0f;
    float8 acc3 = 0.0f;

    // Linear thread ID within workgroup
    const int local_id = get_local_id(1) * 16 + get_local_id(0);
    // 64 threads, each loads some portion of A tile (32 rows x 32 cols = 1024 halfs)
    // 64 threads, 1024 elements -> 16 elements per thread

    // Cooperative A load: each thread loads 16 half values
    // Thread i loads elements [i*16 .. i*16+15] in row-major order of the 32x32 tile
    // That maps to row = (i*16)/32, col = (i*16)%32 ... but let's do it differently
    // Each thread loads one row partially: 64 threads, 32 rows x 32 cols
    // 2 threads per row, each loads 16 elements
    const int a_load_row = local_id / 2;        // 0..31
    const int a_load_col_base = (local_id % 2) * 16; // 0 or 16

    // Load first A tile into buffer 0
    int buf = 0;
    {
        int a_row = tile_row + a_load_row;
        int a_col = a_load_col_base;
        __global const half* a_ptr = A + a_row * K + a_col;
        __local half* slm_ptr = slm_A + buf * 32 * SLM_STRIDE + a_load_row * SLM_STRIDE + a_load_col_base;

        if (a_row < M) {
            for (int i = 0; i < 16; i++) {
                slm_ptr[i] = a_ptr[i];
            }
        } else {
            for (int i = 0; i < 16; i++) {
                slm_ptr[i] = 0.0h;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Main K-loop, step by 32
    const int k_tiles = K / 32;  // K=2048, so 64 iterations

    for (int kt = 0; kt < k_tiles; kt++) {
        int k_base = kt * 32;
        int next_buf = 1 - buf;

        // Prefetch next A tile into next_buf (if not last iteration)
        if (kt + 1 < k_tiles) {
            int next_k = (kt + 1) * 32;
            int a_row = tile_row + a_load_row;
            __global const half* a_ptr = A + a_row * K + next_k + a_load_col_base;
            __local half* slm_ptr = slm_A + next_buf * 32 * SLM_STRIDE + a_load_row * SLM_STRIDE + a_load_col_base;

            if (a_row < M) {
                for (int i = 0; i < 16; i++) {
                    slm_ptr[i] = a_ptr[i];
                }
            } else {
                for (int i = 0; i < 16; i++) {
                    slm_ptr[i] = 0.0h;
                }
            }
        }

        // Compute: process 32 K elements in two k16 steps
        __local half* a_base = slm_A + buf * 32 * SLM_STRIDE;

        // B pointer for this k-step, this subgroup's 16 columns
        __global const half* b_ptr = B + k_base * N + tile_col;

        for (int kk = 0; kk < 2; kk++) {
            // Load B tile: 16 rows x 16 cols -> int8 per WI (VNNI packed)
            // B is row-major [K, N]. We need 16 consecutive K rows, 16 cols.
            // For DPAS b operand (int8): each int holds 2 half values (k-pair)
            // Layout: 8 pairs of k × 16 cols. Each WI holds one column.
            // int8 b means 8 ints = 8 k-pairs = 16 k-values for one column
            int8 b_val;
            __global const half* b_col = b_ptr + kk * 16 * N + sg_lid;
            b_val.s0 = *((__global const int*)(b_ptr + (kk * 16 + 0) * N + tile_col) + sg_lid) ;

            // Actually, let me load B properly for DPAS
            // DPAS b operand int8: represents 16K x 16N tile in VNNI format
            // VNNI: pairs of K are packed into int (2 halfs per int)
            // So int8 per WI: 8 k-pairs, WI lane = column index
            // b[wi] = { pack(B[k+0,col], B[k+1,col]), pack(B[k+2,col], B[k+3,col]), ... }

            // Load B: each WI loads its column, 16 K values packed as 8 ints
            int8 b_data;
            for (int i = 0; i < 8; i++) {
                half b0 = *(b_ptr + (kk * 16 + i * 2) * N + tile_col + sg_lid);
                half b1 = *(b_ptr + (kk * 16 + i * 2 + 1) * N + tile_col + sg_lid);
                // Pack two halfs into one int
                ushort u0 = as_ushort(b0);
                ushort u1 = as_ushort(b1);
                ((int*)&b_data)[i] = (int)u0 | ((int)u1 << 16);
            }

            // Load A from SLM: 32 rows x 16 cols for this k16 step
            // DPAS a operand (short8): 8 rows x 16 K values packed as short8
            // Each short holds one half value? No - short8 = 8 shorts = 8 half values
            // Actually for f16 DPAS with k16: a operand is short8 representing 8 rows
            // Each row needs 16 half values = 16 shorts... that's more than short8
            // 
            // Let me reconsider: intel_sub_group_f16_f16_matrix_mad_k16(short8 a, int8 b, float8 acc)
            // short8 a: across the subgroup (16 WIs), represents 8 rows x 16 K
            //   Each WI holds 8 shorts = 8 rows, one K-element per row for that WI's K-lane
            //   Wait no - with exec_size=16 and k=16: 
            //   a is distributed: 16 WIs × short8 = 16×8 shorts = 128 shorts
            //   This represents 8 rows × 16 K-elements = 128 half values. Yes!
            //   So WI[j] holds a[row][j] for rows 0..7 (the j-th K element of each row)

            // Load A[row_offset..row_offset+7][k_offset..k_offset+15] from SLM
            // For each of 4 groups of 8 rows:
            short8 a0, a1, a2, a3;

            // Row group 0: rows 0-7
            for (int r = 0; r < 8; r++) {
                ((short*)&a0)[r] = as_short(a_base[(0 + r) * SLM_STRIDE + kk * 16 + sg_lid]);
            }
            // Row group 1: rows 8-15
            for (int r = 0; r < 8; r++) {
                ((short*)&a1)[r] = as_short(a_base[(8 + r) * SLM_STRIDE + kk * 16 + sg_lid]);
            }
            // Row group 2: rows 16-23
            for (int r = 0; r < 8; r++) {
                ((short*)&a2)[r] = as_short(a_base[(16 + r) * SLM_STRIDE + kk * 16 + sg_lid]);
            }
            // Row group 3: rows 24-31
            for (int r = 0; r < 8; r++) {
                ((short*)&a3)[r] = as_short(a_base[(24 + r) * SLM_STRIDE + kk * 16 + sg_lid]);
            }

            // DPAS calls
            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b_data, acc0);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b_data, acc1);
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b_data, acc2);
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b_data, acc3);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
        buf = next_buf;
    }

    // Store results
    // acc0: rows 0-7, acc1: rows 8-15, acc2: rows 16-23, acc3: rows 24-31
    // Each float8 has 8 values for 8 rows, column = tile_col + sg_lid
    int out_col = tile_col + sg_lid;
    if (out_col < N) {
        for (int r = 0; r < 8; r++) {
            int out_row = tile_row + r;
            if (out_row < M)
                C[out_row * N + out_col] = convert_half(((float*)&acc0)[r]);
        }
        for (int r = 0; r < 8; r++) {
            int out_row = tile_row + 8 + r;
            if (out_row < M)
                C[out_row * N + out_col] = convert_half(((float*)&acc1)[r]);
        }
        for (int r = 0; r < 8; r++) {
            int out_row = tile_row + 16 + r;
            if (out_row < M)
                C[out_row * N + out_col] = convert_half(((float*)&acc2)[r]);
        }
        for (int r = 0; r < 8; r++) {
            int out_row = tile_row + 24 + r;
            if (out_row < M)
                C[out_row * N + out_col] = convert_half(((float*)&acc3)[r]);
        }
    }
}
```

### Version 2 (Result: Correctness score: 0 / 5 (compilation error), runtime in ms: -1.000):
```OCL
// GEMM: C[M,N] = A[M,K] x B[K,N], row-major, FP16 in/out, FP32 accumulation
// Launch: GWS = ((N/64)*64, M/32), LWS = (64, 1), subgroup_size = 16
// Architecture: 4 subgroups per WG, A in SLM (double-buffered), B from global/L2
// Tile: 32x64x32, K-loop 2x unrolled (step=64)
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
    // Tile indices
    const int tileN = get_group_id(0);  // which 64-col tile
    const int tileM = get_group_id(1);  // which 32-row tile

    const int lid = get_local_id(0);
    const int sg_id = lid / 16;         // subgroup index 0..3
    const int sg_lid = lid % 16;        // lane within subgroup

    // Base row/col for this WG's output tile
    const int baseM = tileM * 32;
    const int baseN = tileN * 64;

    // Each subgroup handles 16 columns: sg_id * 16
    const int sgBaseN = baseN + sg_id * 16;

    // SLM for A: double-buffered, 32 rows x 32 cols (with padding for bank conflicts)
    // Stride = 34 half elements (32 + 2 padding) to avoid bank conflicts
    #define SLM_STRIDE 34
    __local half slm_A[2 * 32 * SLM_STRIDE];  // double buffer

    // Accumulators: 32 rows x 16 cols per subgroup = 4 blocks of 8x16
    // Each DPAS produces float8 (8 rows for this lane's column)
    float8 acc0 = (float8)(0.0f);  // rows 0-7
    float8 acc1 = (float8)(0.0f);  // rows 8-15
    float8 acc2 = (float8)(0.0f);  // rows 16-23
    float8 acc3 = (float8)(0.0f);  // rows 24-31

    // Cooperative A load: 64 WIs load 32x32 = 1024 halfs
    // Each WI loads 1024/64 = 16 halfs
    // Layout: WI i loads elements [i*16 .. i*16+15] in row-major order
    // That's: row = (lid*16) / 32, col_start = (lid*16) % 32
    // Actually let's do: each WI loads one row's worth across 2 iterations
    // Better: 64 WIs, 32 rows, so 2 WIs per row, each loads 16 halfs
    const int a_load_row = lid / 2;        // 0..31
    const int a_load_col = (lid % 2) * 16; // 0 or 16

    // K-loop: step by 64 (2x unrolled), K guaranteed divisible by 64
    // If K not divisible by 64 but by 32, handle last iteration separately
    int k = 0;
    const int K_main = (K / 64) * 64;

    // Prefetch first A tile into SLM buffer 0
    {
        __global const half* a_ptr = A + (baseM + a_load_row) * K + a_load_col;
        __local half* slm_dst = slm_A + a_load_row * SLM_STRIDE + a_load_col;
        // Load 16 halfs
        half8 tmp0 = vload8(0, a_ptr);
        half8 tmp1 = vload8(1, a_ptr);
        vstore8(tmp0, 0, slm_dst);
        vstore8(tmp1, 0, slm_dst + 8);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (; k < K_main; k += 64) {
        // ===== Phase 0: Compute with buffer 0, load A into buffer 1 =====
        {
            // Load next A tile (k+32) into SLM buffer 1
            __global const half* a_ptr = A + (baseM + a_load_row) * K + (k + 32 + a_load_col);
            __local half* slm_dst = slm_A + 32 * SLM_STRIDE + a_load_row * SLM_STRIDE + a_load_col;
            half8 tmp0 = vload8(0, a_ptr);
            half8 tmp1 = vload8(1, a_ptr);

            // Compute from buffer 0: two k16 steps
            __local half* slm_src = slm_A;

            // B pointers for this subgroup
            __global const half* b_ptr = B + k * N + sgBaseN;

            // K-step 0 (k+0..k+15): read A from SLM, B from global
            // DPAS needs: A as short8 (8 rows × k16, packed as pairs), B as int8
            // A layout in SLM: row r, col c -> slm_A[r * SLM_STRIDE + c]
            // For DPAS k16: we need 8 consecutive rows, 16 k-elements each
            // short8 a_tile: each short = 2 half values (crosspack=2)
            // So short8 = 8 shorts = 16 halfs = one row's k16 data... no.
            // Actually for intel_sub_group_f16_f16_matrix_mad_k16:
            //   a: short8 - each WI holds 8 shorts, across 16 WIs this represents 8x16 tile
            //   b: int8 - each WI holds 8 ints, across 16 WIs this represents 16x16 tile
            //   Result: float8 - 8x16 output tile

            // A tile (short8): WI sg_lid reads column sg_lid from 8 rows
            // Each short packs 2 consecutive k values? No - for k16:
            // a[i] for WI w = A[row_base + i][k_base + w] packed as half pairs
            // Actually the exact layout: short8 means 8 elements per WI.
            // For repcount=8, depth=16 (k16): each WI contributes to one column of A
            // a.s0 = pack(A[r0][k_2w], A[r0][k_2w+1]) ... this is crosspack=2
            // Let me use the standard approach: read A from SLM using block reads

            // For A: each subgroup needs 8 rows × 16 k-elements for each DPAS
            // With crosspack=2: stored as [row][k/2] where each element is 2 halfs packed as short
            // But our SLM has standard row-major layout.
            // We'll read A manually: each WI reads from its lane's k-column pair

            // Standard approach for row-major A in SLM with DPAS k16:
            // WI w in subgroup reads: for each of 8 rows, read A[row][2*w] and A[row][2*w+1]
            // packed as short. This gives short8 (8 rows).

            // K-step 0: k offset 0..15
            #define READ_A_BLOCK(row_off, k_off) ({ \
                short8 _a; \
                __local const half* _base = slm_src + (row_off) * SLM_STRIDE + (k_off) + sg_lid * 2; \
                _a.s0 = *((__local const short*)(_base + 0 * SLM_STRIDE)); \
                _a.s1 = *((__local const short*)(_base + 1 * SLM_STRIDE)); \
                _a.s2 = *((__local const short*)(_base + 2 * SLM_STRIDE)); \
                _a.s3 = *((__local const short*)(_base + 3 * SLM_STRIDE)); \
                _a.s4 = *((__local const short*)(_base + 4 * SLM_STRIDE)); \
                _a.s5 = *((__local const short*)(_base + 5 * SLM_STRIDE)); \
                _a.s6 = *((__local const short*)(_base + 6 * SLM_STRIDE)); \
                _a.s7 = *((__local const short*)(_base + 7 * SLM_STRIDE)); \
                _a; \
            })

            // B: 16 k-rows × 16 n-cols, need int8 per WI
            // For DPAS B with k16: int8 means 8 ints = 16 shorts = 16 half-pairs
            // Layout: B[k][n], WI w reads column w. int packs 2 k-values.
            // b.s0 = pack(B[k0][w], B[k1][w]), b.s1 = pack(B[k2][w], B[k3][w]), etc.
            // So int8 = 16 k-values for one n-column.

            #define READ_B_BLOCK(k_off) ({ \
                int8 _b; \
                __global const half* _bbase = b_ptr + (k_off) * N + sg_lid; \
                _b.s0 = *((__global const int*)&(half2){_bbase[0*N], _bbase[1*N]}); \
                _b.s1 = *((__global const int*)&(half2){_bbase[2*N], _bbase[3*N]}); \
                _b.s2 = *((__global const int*)&(half2){_bbase[4*N], _bbase[5*N]}); \
                _b.s3 = *((__global const int*)&(half2){_bbase[6*N], _bbase[7*N]}); \
                _b.s4 = *((__global const int*)&(half2){_bbase[8*N], _bbase[9*N]}); \
                _b.s5 = *((__global const int*)&(half2){_bbase[10*N], _bbase[11*N]}); \
                _b.s6 = *((__global const int*)&(half2){_bbase[12*N], _bbase[13*N]}); \
                _b.s7 = *((__global const int*)&(half2){_bbase[14*N], _bbase[15*N]}); \
                _b; \
            })

            // K-step 0 (k+0..k+15)
            short8 a00 = READ_A_BLOCK(0, 0);
            short8 a10 = READ_A_BLOCK(8, 0);
            short8 a20 = READ_A_BLOCK(16, 0);
            short8 a30 = READ_A_BLOCK(24, 0);

            int8 b0 = READ_B_BLOCK(0);

            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a00, b0, acc0);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a10, b0, acc1);
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a20, b0, acc2);
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a30, b0, acc3);

            // K-step 1 (k+16..k+31)
            short8 a01 = READ_A_BLOCK(0, 16);
            short8 a11 = READ_A_BLOCK(8, 16);
            short8 a21 = READ_A_BLOCK(16, 16);
            short8 a31 = READ_A_BLOCK(24, 16);

            int8 b1 = READ_B_BLOCK(16);

            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a01, b1, acc0);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a11, b1, acc1);
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a21, b1, acc2);
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a31, b1, acc3);

            // Store A to buffer 1
            vstore8(tmp0, 0, slm_dst);
            vstore8(tmp1, 0, slm_dst + 8);
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // ===== Phase 1: Compute with buffer 1, load A into buffer 0 =====
        {
            // Load next A tile (k+64) into SLM buffer 0 (if not last iteration)
            __global const half* a_ptr_next = A + (baseM + a_load_row) * K + (k + 64 + a_load_col);
            __local half* slm_dst_next = slm_A + a_load_row * SLM_STRIDE + a_load_col;
            half8 tmp0_next, tmp1_next;
            if (k + 64 < K) {
                tmp0_next = vload8(0, a_ptr_next);
                tmp1_next = vload8(1, a_ptr_next);
            }

            // Compute from buffer 1
            __local half* slm_src = slm_A + 32 * SLM_STRIDE;
            __global const half* b_ptr = B + (k + 32) * N + sgBaseN;

            // K-step 0 (k+32..k+47)
            short8 a00 = READ_A_BLOCK(0, 0);
            short8 a10 = READ_A_BLOCK(8, 0);
            short8 a20 = READ_A_BLOCK(16, 0);
            short8 a30 = READ_A_BLOCK(24, 0);

            int8 b0 = READ_B_BLOCK(0);

            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a00, b0, acc0);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a10, b0, acc1);
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a20, b0, acc2);
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a30, b0, acc3);

            // K-step 1 (k+48..k+63)
            short8 a01 = READ_A_BLOCK(0, 16);
            short8 a11 = READ_A_BLOCK(8, 16);
            short8 a21 = READ_A_BLOCK(16, 16);
            short8 a31 = READ_A_BLOCK(24, 16);

            int8 b1 = READ_B_BLOCK(16);

            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a01, b1, acc0);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a11, b1, acc1);
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a21, b1, acc2);
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a31, b1, acc3);

            // Store next A to buffer 0
            if (k + 64 < K) {
                vstore8(tmp0_next, 0, slm_dst_next);
                vstore8(tmp1_next, 0, slm_dst_next + 8);
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Handle remaining K (if K not divisible by 64 but divisible by 32)
    if (k < K) {
        // One more 32-step from buffer 0 (already loaded)
        __local half* slm_src = slm_A;
        __global const half* b_ptr = B + k * N + sgBaseN;

        short8 a00 = READ_A_BLOCK(0, 0);
        short8 a10 = READ_A_BLOCK(8, 0);
        short8 a20 = READ_A_BLOCK(16, 0);
        short8 a30 = READ_A_BLOCK(24, 0);
        int8 b0 = READ_B_BLOCK(0);

        acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a00, b0, acc0);
        acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a10, b0, acc1);
        acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a20, b0, acc2);
        acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a30, b0, acc3);

        short8 a01 = READ_A_BLOCK(0, 16);
        short8 a11 = READ_A_BLOCK(8, 16);
        short8 a21 = READ_A_BLOCK(16, 16);
        short8 a31 = READ_A_BLOCK(24, 16);
        int8 b1 = READ_B_BLOCK(16);

        acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a01, b1, acc0);
        acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a11, b1, acc1);
        acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a21, b1, acc2);
        acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a31, b1, acc3);
    }

    #undef READ_A_BLOCK
    #undef READ_B_BLOCK
    #undef SLM_STRIDE

    // Store results: each subgroup writes 32 rows × 16 cols
    // acc0..acc3 contain rows 0-7, 8-15, 16-23, 24-31 for this subgroup's 16 columns
    // float8 acc0: element i = result for (baseM + i, sgBaseN + sg_lid)
    __global half* c_ptr = C + baseM * N + sgBaseN + sg_lid;

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

### Previous kernel and evaluation:
Here is the last OCL kernel we tested (Result: Correctness score: 3 / 5 (compiled but incorrect), runtime in ms: -1.000):
```OCL
// GEMM: C[M,N] = A[M,K] x B[K,N], all half precision, f32 accumulation
// Architecture: 64 WIs (4 SGs of 16), TILE 32x64x32, A in SLM, B from global
// Launch: GWS = (N/16, M/32, 1) * (16, 4, 1), LWS = (16, 4, 1)
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
    // Work-group and subgroup IDs
    const int wg_n = get_group_id(0);  // which 64-col tile
    const int wg_m = get_group_id(1);  // which 32-row tile
    const int sg_id = get_sub_group_id();  // 0..3, each handles 16 columns
    const int sg_lid = get_sub_group_local_id();  // 0..15

    // Global tile offsets
    const int tile_row = wg_m * 32;
    const int tile_col = wg_n * 64 + sg_id * 16;

    // SLM for A: double-buffered, 32 rows x 32 cols with padding
    // Stride = 34 to avoid bank conflicts (34 halfs = 68 bytes per row)
    #define SLM_STRIDE 34
    __local half slm_A[2 * 32 * SLM_STRIDE];  // 2 buffers

    // Accumulators: 32 rows x 16 cols = 4 DPAS results of float8
    float8 acc0 = (float8)(0.0f);
    float8 acc1 = (float8)(0.0f);
    float8 acc2 = (float8)(0.0f);
    float8 acc3 = (float8)(0.0f);

    // Each WI in the WG (64 total) cooperatively loads A tile (32x32 = 1024 halfs)
    // 64 WIs, each loads 1024/64 = 16 halfs per K-step
    const int local_id = sg_id * 16 + sg_lid;  // 0..63

    // Preload first A tile into buffer 0
    {
        // Each WI loads 16 halfs. 1024 halfs / 64 WIs = 16 halfs per WI
        // Map: local_id covers 64 positions, each loads 16 consecutive halfs
        // Total elements = 32*32 = 1024, arranged as 32 rows x 32 cols
        // Each WI handles: row = (local_id * 16) / 32, starting col = (local_id * 16) % 32
        // But simpler: treat as linear, each WI loads elements [local_id*16 .. local_id*16+15]
        // Then store into SLM with proper stride
        int base_elem = local_id * 16;  // 0, 16, 32, ... 1008
        int row = base_elem / 32;       // 0..31
        int col = base_elem % 32;       // 0 or 16

        __global const half* a_ptr = A + (tile_row + row) * K + col;

        // Load 16 halfs from global (contiguous in K dimension for row-major A)
        half16 a_val = vload16(0, a_ptr);

        // Store to SLM buffer 0
        __local half* slm_dst = slm_A + row * SLM_STRIDE + col;
        vstore16(a_val, 0, slm_dst);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int buf = 0;

    // Main K-loop: K/32 iterations
    const int k_iters = K / 32;

    for (int ki = 0; ki < k_iters; ki++) {
        int k_offset = ki * 32;
        int next_k = k_offset + 32;

        // Start loading next A tile into other buffer (if not last iteration)
        // We'll do compute first, then load next (with barrier management)

        // Read A from current SLM buffer and B from global, do DPAS
        __local half* cur_slm = slm_A + buf * 32 * SLM_STRIDE;

        // For DPAS k16: we do 2 DPAS per accumulator (k=0..15, k=16..31)
        // Each DPAS: intel_sub_group_f16_f16_matrix_mad_k16(short8 a, int8 b, float8 acc)
        // a (short8): 8 rows, each row has 16 f16 values packed across the subgroup
        //   - Each WI holds short8 = 8 pairs of f16 (but actually 8 x 1 short = 8 f16 values?)
        //   No: short8 per WI, 16 WIs in SG. Total = 8*16 shorts = 128 shorts = 128 f16 values
        //   That's 8 rows x 16 K elements.
        //   So each WI's short8 contains: for each of 8 rows, one pair of f16 at k-positions [lid*1..?]
        //   Actually: a is 8 rows x 16K. Distributed: each WI holds 8 shorts.
        //   short = 2 f16 packed? No, short = 16 bits = 1 f16.
        //   Wait - intel_sub_group_f16_f16_matrix_mad_k16 with short8 a:
        //   8 rows, k=16. Total a elements = 8*16 = 128 f16 = 128 shorts.
        //   Across 16 WIs: 128/16 = 8 shorts per WI = short8. 
        //   Layout: WI[i] holds a[row][i] for each row packed sequentially.
        //   So a.s0 = row0_k[lid], a.s1 = row1_k[lid], ... a.s7 = row7_k[lid]
        //   Wait no - for k16: each row has 16 k-elements distributed 1 per WI.
        //   So WI[lid] holds: s0=A[row0,k_base+lid], s1=A[row1,k_base+lid], ...

        // Actually for Xe2 DPAS with k16 and f16:
        // a: short8 per WI. 8 = repcount (rows). Each element = 1 f16 at k-position = sg_lid
        // b: int8 per WI. int = 2 f16 packed. 8 ints = 16 f16 values.
        //   b represents 16 K-elements for column = sg_lid (N-direction).
        //   int8 = 16 f16 values for k=0..15 at n=sg_lid

        // Load B: for k16 step, each WI needs int8 = 16 f16 values from B
        // B[k, n]: row-major, so B[k_base + k_idx, tile_col + sg_lid]
        // For 16 consecutive k values at fixed n: stride = N between elements

        // Let's load B for first k16 block
        __global const half* b_base0 = B + k_offset * N + tile_col + sg_lid;
        __global const half* b_base1 = B + (k_offset + 16) * N + tile_col + sg_lid;

        // Load B k=0..15 for this subgroup's 16 columns
        // Each WI loads 16 halfs with stride N
        short b_raw0[16], b_raw1[16];
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            b_raw0[i] = as_short(b_base0[i * N]);
        }
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            b_raw1[i] = as_short(b_base1[i * N]);
        }

        // Pack into int8 for DPAS (pairs of f16 -> int)
        int8 b_val0, b_val1;
        b_val0.s0 = (int)((uint)(ushort)b_raw0[0] | ((uint)(ushort)b_raw0[1] << 16));
        b_val0.s1 = (int)((uint)(ushort)b_raw0[2] | ((uint)(ushort)b_raw0[3] << 16));
        b_val0.s2 = (int)((uint)(ushort)b_raw0[4] | ((uint)(ushort)b_raw0[5] << 16));
        b_val0.s3 = (int)((uint)(ushort)b_raw0[6] | ((uint)(ushort)b_raw0[7] << 16));
        b_val0.s4 = (int)((uint)(ushort)b_raw0[8] | ((uint)(ushort)b_raw0[9] << 16));
        b_val0.s5 = (int)((uint)(ushort)b_raw0[10] | ((uint)(ushort)b_raw0[11] << 16));
        b_val0.s6 = (int)((uint)(ushort)b_raw0[12] | ((uint)(ushort)b_raw0[13] << 16));
        b_val0.s7 = (int)((uint)(ushort)b_raw0[14] | ((uint)(ushort)b_raw0[15] << 16));

        b_val1.s0 = (int)((uint)(ushort)b_raw1[0] | ((uint)(ushort)b_raw1[1] << 16));
        b_val1.s1 = (int)((uint)(ushort)b_raw1[2] | ((uint)(ushort)b_raw1[3] << 16));
        b_val1.s2 = (int)((uint)(ushort)b_raw1[4] | ((uint)(ushort)b_raw1[5] << 16));
        b_val1.s3 = (int)((uint)(ushort)b_raw1[6] | ((uint)(ushort)b_raw1[7] << 16));
        b_val1.s4 = (int)((uint)(ushort)b_raw1[8] | ((uint)(ushort)b_raw1[9] << 16));
        b_val1.s5 = (int)((uint)(ushort)b_raw1[10] | ((uint)(ushort)b_raw1[11] << 16));
        b_val1.s6 = (int)((uint)(ushort)b_raw1[12] | ((uint)(ushort)b_raw1[13] << 16));
        b_val1.s7 = (int)((uint)(ushort)b_raw1[14] | ((uint)(ushort)b_raw1[15] << 16));

        // Load A from SLM for 4 groups of 8 rows, k=0..15
        // A in SLM: row r, col c -> slm[r * SLM_STRIDE + c]
        // For DPAS a (short8): WI[lid] needs A[row_base+r, k_base+lid] for r=0..7
        // k_base=0 for first k16, k_base=16 for second k16

        short8 a_val00, a_val01, a_val02, a_val03;
        short8 a_val10, a_val11, a_val12, a_val13;

        // First k16 block (k=0..15 within the 32-wide tile)
        #pragma unroll
        for (int r = 0; r < 8; r++) {
            ((short*)&a_val00)[r] = as_short(cur_slm[(0 + r) * SLM_STRIDE + sg_lid]);
        }
        #pragma unroll
        for (int r = 0; r < 8; r++) {
            ((short*)&a_val01)[r] = as_short(cur_slm[(8 + r) * SLM_STRIDE + sg_lid]);
        }
        #pragma unroll
        for (int r = 0; r < 8; r++) {
            ((short*)&a_val02)[r] = as_short(cur_slm[(16 + r) * SLM_STRIDE + sg_lid]);
        }
        #pragma unroll
        for (int r = 0; r < 8; r++) {
            ((short*)&a_val03)[r] = as_short(cur_slm[(24 + r) * SLM_STRIDE + sg_lid]);
        }

        // Second k16 block (k=16..31)
        #pragma unroll
        for (int r = 0; r < 8; r++) {
            ((short*)&a_val10)[r] = as_short(cur_slm[(0 + r) * SLM_STRIDE + 16 + sg_lid]);
        }
        #pragma unroll
        for (int r = 0; r < 8; r++) {
            ((short*)&a_val11)[r] = as_short(cur_slm[(8 + r) * SLM_STRIDE + 16 + sg_lid]);
        }
        #pragma unroll
        for (int r = 0; r < 8; r++) {
            ((short*)&a_val12)[r] = as_short(cur_slm[(16 + r) * SLM_STRIDE + 16 + sg_lid]);
        }
        #pragma unroll
        for (int r = 0; r < 8; r++) {
            ((short*)&a_val13)[r] = as_short(cur_slm[(24 + r) * SLM_STRIDE + 16 + sg_lid]);
        }

        // DPAS calls: 4 row-groups x 2 k-steps = 8 DPAS
        acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a_val00, b_val0, acc0);
        acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a_val01, b_val0, acc1);
        acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a_val02, b_val0, acc2);
        acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a_val03, b_val0, acc3);

        acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a_val10, b_val1, acc0);
        acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a_val11, b_val1, acc1);
        acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a_val12, b_val1, acc2);
        acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a_val13, b_val1, acc3);

        // Load next A tile (if not last iteration)
        if (ki < k_iters - 1) {
            int next_buf = 1 - buf;
            int base_elem = local_id * 16;
            int row = base_elem / 32;
            int col = base_elem % 32;

            __global const half* a_ptr = A + (tile_row + row) * K + next_k + col;
            half16 a_load = vload16(0, a_ptr);

            __local half* slm_dst = slm_A + next_buf * 32 * SLM_STRIDE + row * SLM_STRIDE + col;
            vstore16(a_load, 0, slm_dst);

            buf = next_buf;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write C: 32 rows x 16 cols per subgroup
    // acc0 = rows 0..7, acc1 = rows 8..15, acc2 = rows 16..23, acc3 = rows 24..31
    // Each float8 element corresponds to one row, sg_lid selects column

    #pragma unroll
    for (int r = 0; r < 8; r++) {
        int row_idx = tile_row + r;
        int col_idx = tile_col + sg_lid;
        if (row_idx < M && col_idx < N) {
            C[row_idx * N + col_idx] = convert_half(((float*)&acc0)[r]);
        }
    }
    #pragma unroll
    for (int r = 0; r < 8; r++) {
        int row_idx = tile_row + 8 + r;
        int col_idx = tile_col + sg_lid;
        if (row_idx < M && col_idx < N) {
            C[row_idx * N + col_idx] = convert_half(((float*)&acc1)[r]);
        }
    }
    #pragma unroll
    for (int r = 0; r < 8; r++) {
        int row_idx = tile_row + 16 + r;
        int col_idx = tile_col + sg_lid;
        if (row_idx < M && col_idx < N) {
            C[row_idx * N + col_idx] = convert_half(((float*)&acc2)[r]);
        }
    }
    #pragma unroll
    for (int r = 0; r < 8; r++) {
        int row_idx = tile_row + 24 + r;
        int col_idx = tile_col + sg_lid;
        if (row_idx < M && col_idx < N) {
            C[row_idx * N + col_idx] = convert_half(((float*)&acc3)[r]);
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

self = <task.TestMatmulOCL object at 0x7c152a9f9670>
kernel = <function initialize_matmul_kernel.<locals>.run at 0x7c151c9411c0>
ocl_queue = <pyopencl._cl.CommandQueue object at 0x7c152aa41fd0>, _run = 0

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
E        +  where False = <function allclose at 0x7c1582f60930>(array([[-12.4375   , -85.25     ,  45.875    , ...,   0.       ,\n          0.       ,   0.       ],\n       [-82.9375   ,  28.328125 ,   4.3085938, ...,   0.       ,\n          0.       ,   0.       ],\n       [ 31.09375  , -51.78125  ,  -9.3046875, ...,   0.       ,\n          0.       ,   0.       ],\n       ...,\n       [  0.       ,   0.       ,   0.       , ...,   0.       ,\n          0.       ,   0.       ],\n       [  0.       ,   0.       ,   0.       , ...,   0.       ,\n          0.       ,   0.       ],\n       [  0.       ,   0.       ,   0.       , ...,   0.       ,\n          0.       ,   0.       ]], shape=(2048, 2048), dtype=float32), array([[-12.434087 , -85.22102  ,  45.86866  , ..., -67.074715 ,\n        -64.52674  ,  37.798523 ],\n       [-82.95244  ,  28.332115 ,   4.3084497, ...,  37.17192  ,\n         48.87541  ,  55.1519   ],\n       [ 31.096529 , -51.77693  ,  -9.3054905, ...,   8.124319 ,\n         61.21928  ,   4.7092314],\n       ...,\n       [-65.29967  , -27.73106  ,  74.195465 , ..., 122.09403  ,\n        -41.569603 ,  10.711429 ],\n       [ 44.6838   ,   2.3142765,  22.61605  , ..., -35.807106 ,\n         42.793472 ,  52.60636  ],\n       [ 50.399834 ,  -3.015791 ,  21.545517 , ..., -21.399685 ,\n        -36.035267 ,  49.01544  ]], shape=(2048, 2048), dtype=float32), rtol=0.02, atol=0.02)
E        +    where <function allclose at 0x7c1582f60930> = np.allclose

task.py:244: AssertionError
________________ TestMatmulOCL.test_correctness_wrt_pytorch[1] _________________

self = <task.TestMatmulOCL object at 0x7c152ab1e7e0>
kernel = <function initialize_matmul_kernel.<locals>.run at 0x7c151c9411c0>
ocl_queue = <pyopencl._cl.CommandQueue object at 0x7c152aa41fd0>, _run = 1

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
E        +  where False = <function allclose at 0x7c1582f60930>(array([[ 63.21875  ,  13.6640625,  62.71875  , ...,   0.       ,\n          0.       ,   0.       ],\n       [ 30.34375  ,  -9.578125 , -15.8515625, ...,   0.       ,\n          0.       ,   0.       ],\n       [ -9.828125 ,  26.84375  , -39.875    , ...,   0.       ,\n          0.       ,   0.       ],\n       ...,\n       [  0.       ,   0.       ,   0.       , ...,   0.       ,\n          0.       ,   0.       ],\n       [  0.       ,   0.       ,   0.       , ...,   0.       ,\n          0.       ,   0.       ],\n       [  0.       ,   0.       ,   0.       , ...,   0.       ,\n          0.       ,   0.       ]], shape=(2048, 2048), dtype=float32), array([[  63.220627 ,   13.663691 ,   62.708282 , ...,   26.950535 ,\n        -100.14888  ,  -76.10468  ],\n       [  30.338015 ,   -9.576593 ,  -15.848044 , ...,  -86.66203  ,\n           6.3691177,    9.569207 ],\n       [  -9.825886 ,   26.83852  ,  -39.88768  , ...,   94.32298  ,\n         -40.437588 ,   13.349518 ],\n       ...,\n       [ -50.946926 ,  -10.7210655,  -18.652342 , ...,   -4.0612535,\n         -29.112085 ,   -2.7683525],\n       [ -41.46417  ,   -5.034666 ,   35.500336 , ...,    3.5289268,\n          14.26104  ,   55.58531  ],\n       [ -33.896618 ,   51.45737  ,   13.108513 , ...,   11.92079  ,\n         -64.022385 ,   63.048595 ]], shape=(2048, 2048), dtype=float32), rtol=0.02, atol=0.02)
E        +    where <function allclose at 0x7c1582f60930> = np.allclose

task.py:244: AssertionError
_______________ TestMatmulOCL.test_correctness_wrt_reference[0] ________________

self = <task.TestMatmulOCL object at 0x7c152aa2f8c0>
kernel = <function initialize_matmul_kernel.<locals>.run at 0x7c151c9411c0>
ocl_queue = <pyopencl._cl.CommandQueue object at 0x7c152aa41fd0>, _run = 0

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
E        +  where False = <function allclose at 0x7c1582f60930>(array([[-57.125    , -28.015625 ,  65.5625   , ...,   0.       ,\n          0.       ,   0.       ],\n       [ 92.6875   ,  18.34375  , -14.34375  , ...,   0.       ,\n          0.       ,   0.       ],\n       [ -2.5585938, -30.015625 ,  49.9375   , ...,   0.       ,\n          0.       ,   0.       ],\n       ...,\n       [  0.       ,   0.       ,   0.       , ...,   0.       ,\n          0.       ,   0.       ],\n       [  0.       ,   0.       ,   0.       , ...,   0.       ,\n          0.       ,   0.       ],\n       [  0.       ,   0.       ,   0.       , ...,   0.       ,\n          0.       ,   0.       ]], shape=(2048, 2048), dtype=float32), array([[-5.7125000e+01, -2.8015625e+01,  6.5562500e+01, ...,\n        -5.0218750e+01,  2.7000000e+01,  2.6109375e+01],\n       [ 9.2687500e+01,  1.8343750e+01, -1.4343750e+01, ...,\n        -4.0843750e+01, -2.5828125e+01, -3.1914062e+00],\n       [-2.5585938e+00, -3.0015625e+01,  4.9937500e+01, ...,\n        -5.7156250e+01, -9.2250000e+01,  1.2921875e+01],\n       ...,\n       [-4.6031250e+01, -1.4262500e+02, -6.1500000e+01, ...,\n        -1.2194824e-01,  6.6445312e+00, -5.3156250e+01],\n       [ 3.1859375e+01, -2.3484375e+01,  4.2750000e+01, ...,\n         1.8390625e+01, -2.0507812e+00, -1.6000000e+02],\n       [-3.8968750e+01,  9.1750000e+01,  2.3953125e+01, ...,\n         4.5437500e+01,  5.4437500e+01,  1.0787500e+02]],\n      shape=(2048, 2048), dtype=float32), rtol=0.02, atol=0.02)
E        +    where <function allclose at 0x7c1582f60930> = np.allclose

task.py:262: AssertionError
_______________ TestMatmulOCL.test_correctness_wrt_reference[1] ________________

self = <task.TestMatmulOCL object at 0x7c152aa2faa0>
kernel = <function initialize_matmul_kernel.<locals>.run at 0x7c151c9411c0>
ocl_queue = <pyopencl._cl.CommandQueue object at 0x7c152aa41fd0>, _run = 1

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
E        +  where False = <function allclose at 0x7c1582f60930>(array([[ 46.4375  , -21.75    , -36.71875 , ...,   0.      ,   0.      ,\n          0.      ],\n       [-13.859375, -17.453125,  17.375   , ...,   0.      ,   0.      ,\n          0.      ],\n       [ 34.1875  ,  -9.421875, -49.      , ...,   0.      ,   0.      ,\n          0.      ],\n       ...,\n       [  0.      ,   0.      ,   0.      , ...,   0.      ,   0.      ,\n          0.      ],\n       [  0.      ,   0.      ,   0.      , ...,   0.      ,   0.      ,\n          0.      ],\n       [  0.      ,   0.      ,   0.      , ...,   0.      ,   0.      ,\n          0.      ]], shape=(2048, 2048), dtype=float32), array([[  46.4375   ,  -21.75     ,  -36.71875  , ...,  -39.96875  ,\n           3.125    , -147.25     ],\n       [ -13.859375 ,  -17.453125 ,   17.375    , ...,   57.90625  ,\n          25.859375 ,  -68.0625   ],\n       [  34.1875   ,   -9.421875 ,  -49.       , ...,  -46.71875  ,\n         -64.5625   ,   37.6875   ],\n       ...,\n       [   3.3261719,  -60.03125  ,  -64.       , ...,   18.328125 ,\n         -17.234375 ,  -30.90625  ],\n       [  -0.9145508,   97.25     ,  -18.109375 , ...,   35.75     ,\n         -45.0625   ,   21.78125  ],\n       [ -36.5      ,  -63.8125   ,   68.125    , ...,  -48.09375  ,\n         -29.515625 ,   11.1640625]], shape=(2048, 2048), dtype=float32), rtol=0.02, atol=0.02)
E        +    where <function allclose at 0x7c1582f60930> = np.allclose

task.py:262: AssertionError
=========================== short test summary info ============================
FAILED task.py::TestMatmulOCL::test_correctness_wrt_pytorch[0] - AssertionErr...
FAILED task.py::TestMatmulOCL::test_correctness_wrt_pytorch[1] - AssertionErr...
FAILED task.py::TestMatmulOCL::test_correctness_wrt_reference[0] - AssertionE...
FAILED task.py::TestMatmulOCL::test_correctness_wrt_reference[1] - AssertionE...
======================= 4 failed, 1 deselected in 0.88s ========================

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


[COMPILE ERROR FEEDBACK FROM PREVIOUS TRIAL]
The following compile errors occurred in the previous iteration. Avoid these mistakes in your implementation:
Branch 2: Syntax precheck failed: clBuildProgram failed: BUILD_PROGRAM_FAILURE - clBuildProgram failed: BUILD_PROGRAM_FAILURE - clBuildProgram failed: BUILD_PROGRAM_FAILURE

Build on <pyopencl.Device 'Intel(R) Arc(TM) B580 Graphics' on 'Intel(R) OpenCL Graphics' at 0x5b1b7e7feb90>:

3:154:23: error: casting 'half2 *' to type 'const 
Branch 3: Syntax precheck failed: clBuildProgram failed: BUILD_PROGRAM_FAILURE - clBuildProgram failed: BUILD_PROGRAM_FAILURE - clBuildProgram failed: BUILD_PROGRAM_FAILURE

Build on <pyopencl.Device 'Intel(R) Arc(TM) B580 Graphics' on 'Intel(R) OpenCL Graphics' at 0x5b1b7e7feb90>:

4:12:26: warning: OpenCL extension 'cl_intel_requi
[END COMPILE ERROR FEEDBACK]
