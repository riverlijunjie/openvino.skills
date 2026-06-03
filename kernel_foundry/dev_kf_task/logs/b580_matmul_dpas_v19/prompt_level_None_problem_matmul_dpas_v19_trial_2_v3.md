

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
// GEMM: C[M,N] = A[M,K] x B[K,N], all row-major FP16, f32 accumulation
// Launch: GWS = ((N/64)*64, M/32), LWS = (64, 1)
//   - 64 WIs per WG = 4 subgroups of 16
//   - Each WG computes a 32×64 output tile
//   - TILE_M=32, TILE_N=64, TILE_K=32
//   - A loaded cooperatively into SLM (double-buffered)
//   - B loaded directly from global memory
// Hardware: Intel B580, Xe2, exec_size=16

#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable

#define TILE_M 32
#define TILE_N 64
#define TILE_K 32
#define SG_SIZE 16
#define NUM_SG 4

// SLM stride with padding to avoid bank conflicts
// A tile: 32 rows × 32 cols of half, stored with stride 34 (32 + 2 padding)
#define SLM_A_STRIDE 34

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
    // Work-group tile position
    const int wg_n = get_group_id(0);  // which 64-col tile
    const int wg_m = get_group_id(1);  // which 32-row tile

    const int m_start = wg_m * TILE_M;
    const int n_start = wg_n * TILE_N;

    // Subgroup and lane identification
    const int local_id = get_local_id(0);  // 0..63
    const int sg_id = local_id / SG_SIZE;  // 0..3
    const int sg_lid = local_id % SG_SIZE; // 0..15

    // Each subgroup handles 16 columns of the 64-col output tile
    const int sg_n_offset = sg_id * SG_SIZE;  // 0, 16, 32, 48

    // SLM for A: double buffer, each buffer is TILE_M × SLM_A_STRIDE halfs
    __local half slm_a[2 * TILE_M * SLM_A_STRIDE];

    // Accumulators: 32 rows × 16 cols per subgroup = 4 DPAS blocks of 8 rows each
    float8 acc0 = (float8)(0.0f);  // rows 0-7
    float8 acc1 = (float8)(0.0f);  // rows 8-15
    float8 acc2 = (float8)(0.0f);  // rows 16-23
    float8 acc3 = (float8)(0.0f);  // rows 24-31

    // Cooperative A load mapping:
    // 64 WIs load 32 rows × 32 cols = 1024 halfs
    // Each WI loads 16 halfs (one half-row)
    // local_id / 2 -> row (0..31), local_id % 2 -> which half (0=cols 0-15, 1=cols 16-31)
    const int a_load_row = local_id / 2;       // 0..31
    const int a_load_col = (local_id % 2) * 16; // 0 or 16

    // Load first A tile into SLM buffer 0
    {
        __global const half* a_src = A + (m_start + a_load_row) * K + a_load_col;
        __local half* a_dst = slm_a + a_load_row * SLM_A_STRIDE + a_load_col;

        half8 v0 = vload8(0, a_src);
        half8 v1 = vload8(1, a_src);
        vstore8(v0, 0, a_dst);
        vstore8(v1, 0, a_dst + 8);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const int k_iters = K / TILE_K;

    for (int ki = 0; ki < k_iters; ki++) {
        int cur_buf = ki & 1;
        int next_buf = 1 - cur_buf;
        int k_base = ki * TILE_K;

        // Prefetch next A tile (if not last iteration)
        if (ki < k_iters - 1) {
            int next_k = k_base + TILE_K;
            __global const half* a_src = A + (m_start + a_load_row) * K + next_k + a_load_col;
            __local half* a_dst = slm_a + next_buf * TILE_M * SLM_A_STRIDE + a_load_row * SLM_A_STRIDE + a_load_col;

            half8 v0 = vload8(0, a_src);
            half8 v1 = vload8(1, a_src);
            vstore8(v0, 0, a_dst);
            vstore8(v1, 0, a_dst + 8);
        }

        // Current SLM A base
        __local half* a_slm = slm_a + cur_buf * TILE_M * SLM_A_STRIDE;

        // Process TILE_K=32 as two k16 steps
        #pragma unroll
        for (int kk = 0; kk < 2; kk++) {
            int k_inner = kk * 16;
            int k_global = k_base + k_inner;

            // Load B: 16 k-rows × 16 cols for this subgroup
            // B is row-major [K][N]. Each lane loads its column across 16 k-rows.
            // Pack into int8 (VNNI format): each int = 2 consecutive fp16 k-values
            __global const half* b_col = B + k_global * N + n_start + sg_n_offset + sg_lid;

            int8 b_packed;
            #pragma unroll
            for (int p = 0; p < 8; p++) {
                ushort lo = as_ushort(b_col[(2 * p) * N]);
                ushort hi = as_ushort(b_col[(2 * p + 1) * N]);
                ((int*)&b_packed)[p] = ((int)hi << 16) | (int)lo;
            }

            // Load A from SLM and compute DPAS for each 8-row block
            // For intel_sub_group_f16_f16_matrix_mad_k16 on Xe2 (exec_size=16):
            //   a: short8 - represents 8×16 matrix A[8 rows][16 k-cols]
            //     Each lane l holds A[row][l] as a short (reinterpreted half)
            //     So a[r] = as_short(A[base_row + r][k_inner + sg_lid])
            //   b: int8 - represents 16×16 matrix B[16 k-rows][16 cols] in VNNI
            //     Each lane l holds column l, pairs of k packed as int
            //   Result: float8 = 8 output rows × 1 col (the lane's column)

            // Block 0: rows 0-7
            short8 a0;
            #pragma unroll
            for (int r = 0; r < 8; r++) {
                a0[r] = as_short(a_slm[(0 + r) * SLM_A_STRIDE + k_inner + sg_lid]);
            }
            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b_packed, acc0);

            // Block 1: rows 8-15
            short8 a1;
            #pragma unroll
            for (int r = 0; r < 8; r++) {
                a1[r] = as_short(a_slm[(8 + r) * SLM_A_STRIDE + k_inner + sg_lid]);
            }
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b_packed, acc1);

            // Block 2: rows 16-23
            short8 a2;
            #pragma unroll
            for (int r = 0; r < 8; r++) {
                a2[r] = as_short(a_slm[(16 + r) * SLM_A_STRIDE + k_inner + sg_lid]);
            }
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b_packed, acc2);

            // Block 3: rows 24-31
            short8 a3;
            #pragma unroll
            for (int r = 0; r < 8; r++) {
                a3[r] = as_short(a_slm[(24 + r) * SLM_A_STRIDE + k_inner + sg_lid]);
            }
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b_packed, acc3);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write results: each lane writes one column across 32 rows
    __global half* c_out = C + m_start * N + n_start + sg_n_offset + sg_lid;

    #pragma unroll
    for (int r = 0; r < 8; r++) {
        c_out[(0 + r) * N] = convert_half(acc0[r]);
    }
    #pragma unroll
    for (int r = 0; r < 8; r++) {
        c_out[(8 + r) * N] = convert_half(acc1[r]);
    }
    #pragma unroll
    for (int r = 0; r < 8; r++) {
        c_out[(16 + r) * N] = convert_half(acc2[r]);
    }
    #pragma unroll
    for (int r = 0; r < 8; r++) {
        c_out[(24 + r) * N] = convert_half(acc3[r]);
    }
}
```

### Version 2 (Result: Correctness score: 3 / 5 (compiled but incorrect), runtime in ms: -1.000):
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

### Previous kernel and evaluation:
Here is the last OCL kernel we tested (Result: Correctness score: 5 / 5 (compiles and correct), runtime in ms: 1.500):
```OCL
// GEMM: C[M,N] = A[M,K] x B[K,N], all row-major FP16, f32 accumulation
// Launch: GWS = (N, M/32), LWS = (64, 1)
//   - 64 WIs per WG = 4 subgroups of 16
//   - Each WG computes a 32×64 output tile
//   - TILE_M=32, TILE_N=64, TILE_K=32
// A is loaded cooperatively into SLM (double-buffered), B from global/L2

#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable

#define TILE_M 32
#define TILE_N 64
#define TILE_K 32
#define SG_SIZE 16
#define NUM_SG 4
// SLM stride with padding to avoid bank conflicts (32 + 4 = 36 halfs = 72 bytes)
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
    // Work-group tile position
    const int wg_n = get_group_id(0);  // which 64-col tile
    const int wg_m = get_group_id(1);  // which 32-row tile

    const int m_start = wg_m * TILE_M;
    const int n_start = wg_n * TILE_N;

    // Subgroup and lane identification
    const int local_id = get_local_id(0);  // 0..63
    const int sg_id = local_id / SG_SIZE;  // 0..3
    const int sg_lid = local_id % SG_SIZE; // 0..15

    // Each subgroup handles 16 columns of the 64-col output tile
    const int sg_n_offset = sg_id * SG_SIZE;

    // SLM double buffer for A: 2 buffers × 32 rows × SLM_A_STRIDE halfs
    __local half slm_a[2 * TILE_M * SLM_A_STRIDE];

    // Accumulators: 32 rows × 16 cols per subgroup = 4 DPAS blocks of 8 rows each
    float8 acc0 = (float8)(0.0f);  // rows 0-7
    float8 acc1 = (float8)(0.0f);  // rows 8-15
    float8 acc2 = (float8)(0.0f);  // rows 16-23
    float8 acc3 = (float8)(0.0f);  // rows 24-31

    // Cooperative A load mapping:
    // 64 threads load 32 rows × 32 cols = 1024 halfs
    // Each thread loads 16 halfs (one half-row)
    // Thread i: row = i/2, col_offset = (i%2)*16
    const int a_load_row = local_id / 2;       // 0..31
    const int a_load_col_half = local_id % 2;  // 0 or 1
    const int a_load_col = a_load_col_half * 16;

    // Load first A tile (k=0) into SLM buffer 0
    {
        __global const half* a_src = A + (m_start + a_load_row) * K + a_load_col;
        __local half* a_dst = slm_a + a_load_row * SLM_A_STRIDE + a_load_col;
        half8 v0 = vload8(0, a_src);
        half8 v1 = vload8(1, a_src);
        vstore8(v0, 0, a_dst);
        vstore8(v1, 0, a_dst + 8);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const int k_iters = K / TILE_K;

    for (int ki = 0; ki < k_iters; ki++) {
        int cur_buf = ki & 1;
        int next_buf = 1 - cur_buf;

        // Prefetch next A tile into next SLM buffer (overlap with compute)
        if (ki < k_iters - 1) {
            int next_k = (ki + 1) * TILE_K;
            __global const half* a_src = A + (m_start + a_load_row) * K + next_k + a_load_col;
            __local half* a_dst = slm_a + next_buf * TILE_M * SLM_A_STRIDE + a_load_row * SLM_A_STRIDE + a_load_col;
            half8 v0 = vload8(0, a_src);
            half8 v1 = vload8(1, a_src);
            vstore8(v0, 0, a_dst);
            vstore8(v1, 0, a_dst + 8);
        }

        // Current A SLM base
        __local half* a_base = slm_a + cur_buf * TILE_M * SLM_A_STRIDE;

        // Current K offset for B
        int k_base = ki * TILE_K;

        // Process TILE_K=32 as two k16 steps (matching DPAS k16)
        #pragma unroll
        for (int kk = 0; kk < 2; kk++) {
            int k_off = k_base + kk * 16;

            // Load B: 16 k-rows × 16 cols for this subgroup
            // B is row-major [K][N]. Each lane loads its column across 16 k-rows.
            // DPAS b operand (int8): 8 ints per lane, each int = 2 packed fp16 (VNNI k-pairs)
            __global const half* b_ptr = B + k_off * N + n_start + sg_n_offset + sg_lid;

            int8 b_val;
            #pragma unroll
            for (int p = 0; p < 8; p++) {
                ushort u0 = as_ushort(b_ptr[(2 * p) * N]);
                ushort u1 = as_ushort(b_ptr[(2 * p + 1) * N]);
                ((int*)&b_val)[p] = (int)((uint)u0 | ((uint)u1 << 16));
            }

            // Load A from SLM and compute DPAS for each 8-row block
            // DPAS a operand (short8): for Xe2 exec_size=16, repcount=8, sdepth=8 (k16 for fp16)
            // Each lane l holds A[row_base + r][k_inner + l] as short (= as_short(half))
            // short8 = 8 rows, one element per row per lane
            int k_inner = kk * 16;

            // Block 0: rows 0-7
            {
                short8 a_val;
                #pragma unroll
                for (int r = 0; r < 8; r++) {
                    a_val[r] = as_short(a_base[r * SLM_A_STRIDE + k_inner + sg_lid]);
                }
                acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a_val, b_val, acc0);
            }

            // Block 1: rows 8-15
            {
                short8 a_val;
                #pragma unroll
                for (int r = 0; r < 8; r++) {
                    a_val[r] = as_short(a_base[(8 + r) * SLM_A_STRIDE + k_inner + sg_lid]);
                }
                acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a_val, b_val, acc1);
            }

            // Block 2: rows 16-23
            {
                short8 a_val;
                #pragma unroll
                for (int r = 0; r < 8; r++) {
                    a_val[r] = as_short(a_base[(16 + r) * SLM_A_STRIDE + k_inner + sg_lid]);
                }
                acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a_val, b_val, acc2);
            }

            // Block 3: rows 24-31
            {
                short8 a_val;
                #pragma unroll
                for (int r = 0; r < 8; r++) {
                    a_val[r] = as_short(a_base[(24 + r) * SLM_A_STRIDE + k_inner + sg_lid]);
                }
                acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a_val, b_val, acc3);
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write results: each lane writes one column across 32 rows
    __global half* c_ptr = C + m_start * N + n_start + sg_n_offset + sg_lid;

    #pragma unroll
    for (int r = 0; r < 8; r++) {
        c_ptr[r * N] = convert_half(acc0[r]);
    }
    #pragma unroll
    for (int r = 0; r < 8; r++) {
        c_ptr[(8 + r) * N] = convert_half(acc1[r]);
    }
    #pragma unroll
    for (int r = 0; r < 8; r++) {
        c_ptr[(16 + r) * N] = convert_half(acc2[r]);
    }
    #pragma unroll
    for (int r = 0; r < 8; r++) {
        c_ptr[(24 + r) * N] = convert_half(acc3[r]);
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

======================= 4 passed, 1 deselected in 0.85s ========================
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
2. Memory Alignment: Align data structures to vector boundaries. Use __attribute__((aligned(N))) where N is 16, 32, or 64. Ensure global memory allocations are aligned for coalesced access.

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