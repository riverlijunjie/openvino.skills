

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

## Best implementation so far:

This is the code of the best kernel we have tested so far (Result: Correctness score: 5 / 5 (compiles and correct), runtime in ms: 1.500):
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

### Previous kernel and evaluation:
Here is the last OCL kernel we tested (Result: Correctness score: 3 / 5 (compiled but incorrect), runtime in ms: -1.000):
```OCL
// GEMM: C[M,N] = A[M,K] x B[K,N], all row-major FP16, f32 accumulation
// Architecture: 64 WIs per WG (4 subgroups of 16), A in SLM, B from global
// Each WG computes a 32×64 output tile (TILE_M=32, TILE_N=64, TILE_K=32)
// Launch: GWS = ((N/64)*4, M/32, 1), LWS = (4, 1, 1)
//   - reqd_work_group_size(4,1,1) with intel_reqd_sub_group_size(16)
//   - This gives 4 subgroups × 16 lanes = 64 work-items per WG
//   - Each subgroup handles 16 columns of the 64-column tile

#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable

#define TILE_M 32
#define TILE_N 64
#define TILE_K 32
#define SG_SIZE 16
#define NUM_SG 4
// SLM stride with padding to avoid bank conflicts (32 + 4 = 36 halfs = 72 bytes)
#define SLM_STRIDE 36

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
    // Work-group identification
    const int wg_n = get_group_id(0);  // which 64-col tile
    const int wg_m = get_group_id(1);  // which 32-row tile

    // Subgroup identification
    const int sg_id = get_sub_group_id();        // 0..3
    const int sg_lid = get_sub_group_local_id(); // 0..15

    // Global tile offsets
    const int m_start = wg_m * TILE_M;
    const int n_start = wg_n * TILE_N;

    // Each subgroup handles 16 columns
    const int sg_col_offset = sg_id * SG_SIZE;

    // Linear local ID for cooperative loading
    const int local_id = sg_id * SG_SIZE + sg_lid;  // 0..63

    // SLM double buffer for A: 2 buffers × 32 rows × SLM_STRIDE halfs
    __local half slm_a[2 * TILE_M * SLM_STRIDE];

    // Accumulators: 32 rows × 16 cols per subgroup = 4 DPAS blocks of 8 rows each
    float8 acc0 = (float8)(0.0f);  // rows 0-7
    float8 acc1 = (float8)(0.0f);  // rows 8-15
    float8 acc2 = (float8)(0.0f);  // rows 16-23
    float8 acc3 = (float8)(0.0f);  // rows 24-31

    // Cooperative A load mapping:
    // 64 threads load 32 rows × 32 cols = 1024 halfs
    // Each thread loads 16 halfs (one half-row)
    // Thread i loads row (i/2), columns (i%2)*16 .. (i%2)*16+15
    const int a_load_row = local_id / 2;
    const int a_load_col_half = local_id % 2;  // 0 or 1
    const int a_load_col_base = a_load_col_half * 16;

    // Load first A tile into SLM buffer 0
    {
        __global const half* a_src = A + (m_start + a_load_row) * K + a_load_col_base;
        __local half* slm_dst = slm_a + a_load_row * SLM_STRIDE + a_load_col_base;
        // Load 16 halfs (256 bits = 32 bytes)
        half8 v0 = vload8(0, a_src);
        half8 v1 = vload8(1, a_src);
        vstore8(v0, 0, slm_dst);
        vstore8(v1, 0, slm_dst + 8);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const int k_iters = K / TILE_K;  // K=2048, TILE_K=32 -> 64 iterations

    for (int ki = 0; ki < k_iters; ki++) {
        int cur_buf = ki & 1;
        int next_buf = 1 - cur_buf;
        int k_base = ki * TILE_K;

        // Prefetch next A tile into next SLM buffer (overlap with compute)
        if (ki < k_iters - 1) {
            int next_k = k_base + TILE_K;
            __global const half* a_src = A + (m_start + a_load_row) * K + next_k + a_load_col_base;
            __local half* slm_dst = slm_a + next_buf * TILE_M * SLM_STRIDE + a_load_row * SLM_STRIDE + a_load_col_base;
            half8 v0 = vload8(0, a_src);
            half8 v1 = vload8(1, a_src);
            vstore8(v0, 0, slm_dst);
            vstore8(v1, 0, slm_dst + 8);
        }

        // Current SLM buffer base
        __local const half* a_slm = slm_a + cur_buf * TILE_M * SLM_STRIDE;

        // B base for this k-step and this subgroup's columns
        __global const half* b_base = B + k_base * N + n_start + sg_col_offset;

        // Process TILE_K=32 as two k16 steps
        #pragma unroll
        for (int kk = 0; kk < 2; kk++) {
            int k_inner = kk * 16;

            // Load B: 16 k-rows × 16 cols in VNNI format (int8 per WI)
            // Each WI (lane l) loads column l, packing pairs of k-rows into ints
            // b[p] = pack(B[k+2p][col+l], B[k+2p+1][col+l])
            int8 b_packed;
            __global const half* b_ptr = b_base + k_inner * N + sg_lid;

            #pragma unroll
            for (int p = 0; p < 8; p++) {
                ushort lo = as_ushort(b_ptr[(2 * p) * N]);
                ushort hi = as_ushort(b_ptr[(2 * p + 1) * N]);
                ((int*)&b_packed)[p] = ((int)hi << 16) | (int)lo;
            }

            // For each 8-row block, load A from SLM and do DPAS
            // A operand for intel_sub_group_f16_f16_matrix_mad_k16:
            //   short8 per WI: a[r] = as_short(A[row+r][k_base + k_inner + lane])
            //   Each lane l holds one k-element per row

            // Block 0: rows 0-7
            {
                short8 a_val;
                #pragma unroll
                for (int r = 0; r < 8; r++) {
                    ((short*)&a_val)[r] = as_short(a_slm[r * SLM_STRIDE + k_inner + sg_lid]);
                }
                acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a_val, b_packed, acc0);
            }

            // Block 1: rows 8-15
            {
                short8 a_val;
                #pragma unroll
                for (int r = 0; r < 8; r++) {
                    ((short*)&a_val)[r] = as_short(a_slm[(8 + r) * SLM_STRIDE + k_inner + sg_lid]);
                }
                acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a_val, b_packed, acc1);
            }

            // Block 2: rows 16-23
            {
                short8 a_val;
                #pragma unroll
                for (int r = 0; r < 8; r++) {
                    ((short*)&a_val)[r] = as_short(a_slm[(16 + r) * SLM_STRIDE + k_inner + sg_lid]);
                }
                acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a_val, b_packed, acc2);
            }

            // Block 3: rows 24-31
            {
                short8 a_val;
                #pragma unroll
                for (int r = 0; r < 8; r++) {
                    ((short*)&a_val)[r] = as_short(a_slm[(24 + r) * SLM_STRIDE + k_inner + sg_lid]);
                }
                acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a_val, b_packed, acc3);
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write results: each WI writes one column across 32 rows
    // acc[i] contains the result for row (block*8 + i), column (n_start + sg_col_offset + sg_lid)
    __global half* c_out = C + m_start * N + n_start + sg_col_offset + sg_lid;

    #pragma unroll
    for (int r = 0; r < 8; r++) {
        c_out[r * N] = convert_half(((float*)&acc0)[r]);
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

Console output from running this kernel:

Test result on platform Intel Corporation Battlemage G21 [Intel Graphics]:
==== test session starts

task.py::TestMatmulOCL::test_correctness_wrt_pytorch[0] FAILED           [ 25%]
task.py::TestMatmulOCL::test_correctness_wrt_pytorch[1] FAILED           [ 50%]
task.py::TestMatmulOCL::test_correctness_wrt_reference[0] FAILED         [ 75%]
task.py::TestMatmulOCL::test_correctness_wrt_reference[1] FAILED         [100%]

=================================== FAILURES ===================================
________________ TestMatmulOCL.test_correctness_wrt_pytorch[0] _________________

self = <task.TestMatmulOCL object at 0x706315980890>
kernel = <function initialize_matmul_kernel.<locals>.run at 0x7062afd41440>
ocl_queue = <pyopencl._cl.CommandQueue object at 0x7062bdb45f30>, _run = 0

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
E        +  where False = <function allclose at 0x706316368d70>(array([[15.3515625, 26.609375 , -6.2929688, ...,  0.       ,  0.       ,\n         0.       ],\n       [-3.234375 , 43.1875   , 16.75     , ...,  0.       ,  0.       ,\n         0.       ],\n       [       nan,        nan,        nan, ...,  0.       ,  0.       ,\n         0.       ],\n       ...,\n       [ 0.       ,  0.       ,  0.       , ...,  0.       ,  0.       ,\n         0.       ],\n       [       nan,        nan,        nan, ...,  0.       ,  0.       ,\n         0.       ],\n       [ 0.       ,  0.       ,  0.       , ...,  0.       ,  0.       ,\n         0.       ]], shape=(2048, 2048), dtype=float32), array([[-12.434087 , -85.22102  ,  45.86866  , ..., -67.074715 ,\n        -64.52674  ,  37.798523 ],\n       [-82.95244  ,  28.332115 ,   4.3084497, ...,  37.17192  ,\n         48.87541  ,  55.1519   ],\n       [ 31.096529 , -51.77693  ,  -9.3054905, ...,   8.124319 ,\n         61.21928  ,   4.7092314],\n       ...,\n       [-65.29967  , -27.73106  ,  74.195465 , ..., 122.09403  ,\n        -41.569603 ,  10.711429 ],\n       [ 44.6838   ,   2.3142765,  22.61605  , ..., -35.807106 ,\n         42.793472 ,  52.60636  ],\n       [ 50.399834 ,  -3.015791 ,  21.545517 , ..., -21.399685 ,\n        -36.035267 ,  49.01544  ]], shape=(2048, 2048), dtype=float32), rtol=0.02, atol=0.02)
E        +    where <function allclose at 0x706316368d70> = np.allclose

task.py:244: AssertionError
________________ TestMatmulOCL.test_correctness_wrt_pytorch[1] _________________

self = <task.TestMatmulOCL object at 0x7062bdafe360>
kernel = <function initialize_matmul_kernel.<locals>.run at 0x7062afd41440>
ocl_queue = <pyopencl._cl.CommandQueue object at 0x7062bdb45f30>, _run = 1

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
E        +  where False = <function allclose at 0x706316368d70>(array([[-8.4140625, -6.9882812, -0.5864258, ...,  0.       ,  0.       ,\n         0.       ],\n       [ 2.0097656, -2.453125 , -1.8876953, ...,  0.       ,  0.       ,\n         0.       ],\n       [       nan,        nan,        nan, ...,  0.       ,  0.       ,\n         0.       ],\n       ...,\n       [ 0.       ,  0.       ,  0.       , ...,  0.       ,  0.       ,\n         0.       ],\n       [       nan,        nan,        nan, ...,  0.       ,  0.       ,\n         0.       ],\n       [ 0.       ,  0.       ,  0.       , ...,  0.       ,  0.       ,\n         0.       ]], shape=(2048, 2048), dtype=float32), array([[  63.220627 ,   13.663691 ,   62.708282 , ...,   26.950535 ,\n        -100.14888  ,  -76.10468  ],\n       [  30.338015 ,   -9.576593 ,  -15.848044 , ...,  -86.66203  ,\n           6.3691177,    9.569207 ],\n       [  -9.825886 ,   26.83852  ,  -39.88768  , ...,   94.32298  ,\n         -40.437588 ,   13.349518 ],\n       ...,\n       [ -50.946926 ,  -10.7210655,  -18.652342 , ...,   -4.0612535,\n         -29.112085 ,   -2.7683525],\n       [ -41.46417  ,   -5.034666 ,   35.500336 , ...,    3.5289268,\n          14.26104  ,   55.58531  ],\n       [ -33.896618 ,   51.45737  ,   13.108513 , ...,   11.92079  ,\n         -64.022385 ,   63.048595 ]], shape=(2048, 2048), dtype=float32), rtol=0.02, atol=0.02)
E        +    where <function allclose at 0x706316368d70> = np.allclose

task.py:244: AssertionError
_______________ TestMatmulOCL.test_correctness_wrt_reference[0] ________________

self = <task.TestMatmulOCL object at 0x7062bdb2f830>
kernel = <function initialize_matmul_kernel.<locals>.run at 0x7062afd41440>
ocl_queue = <pyopencl._cl.CommandQueue object at 0x7062bdb45f30>, _run = 0

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
E        +  where False = <function allclose at 0x706316368d70>(array([[  4.5039062,  30.140625 ,  14.1484375, ...,   0.       ,\n          0.       ,   0.       ],\n       [ 17.25     , -13.       ,  57.125    , ...,   0.       ,\n          0.       ,   0.       ],\n       [        nan,         nan,         nan, ...,   0.       ,\n          0.       ,   0.       ],\n       ...,\n       [  0.       ,   0.       ,   0.       , ...,   0.       ,\n          0.       ,   0.       ],\n       [        nan,         nan,         nan, ...,   0.       ,\n          0.       ,   0.       ],\n       [  0.       ,   0.       ,   0.       , ...,   0.       ,\n          0.       ,   0.       ]], shape=(2048, 2048), dtype=float32), array([[-5.7125000e+01, -2.8015625e+01,  6.5562500e+01, ...,\n        -5.0218750e+01,  2.7000000e+01,  2.6109375e+01],\n       [ 9.2687500e+01,  1.8343750e+01, -1.4343750e+01, ...,\n        -4.0843750e+01, -2.5828125e+01, -3.1914062e+00],\n       [-2.5585938e+00, -3.0015625e+01,  4.9937500e+01, ...,\n        -5.7156250e+01, -9.2250000e+01,  1.2921875e+01],\n       ...,\n       [-4.6031250e+01, -1.4262500e+02, -6.1500000e+01, ...,\n        -1.2194824e-01,  6.6445312e+00, -5.3156250e+01],\n       [ 3.1859375e+01, -2.3484375e+01,  4.2750000e+01, ...,\n         1.8390625e+01, -2.0507812e+00, -1.6000000e+02],\n       [-3.8968750e+01,  9.1750000e+01,  2.3953125e+01, ...,\n         4.5437500e+01,  5.4437500e+01,  1.0787500e+02]],\n      shape=(2048, 2048), dtype=float32), rtol=0.02, atol=0.02)
E        +    where <function allclose at 0x706316368d70> = np.allclose

task.py:262: AssertionError
_______________ TestMatmulOCL.test_correctness_wrt_reference[1] ________________

self = <task.TestMatmulOCL object at 0x7062bdb2f950>
kernel = <function initialize_matmul_kernel.<locals>.run at 0x7062afd41440>
ocl_queue = <pyopencl._cl.CommandQueue object at 0x7062bdb45f30>, _run = 1

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
E        +  where False = <function allclose at 0x706316368d70>(array([[ 29.53125 , -40.25    , -21.875   , ...,   0.      ,   0.      ,\n          0.      ],\n       [ -4.1875  , -27.828125, -23.46875 , ...,   0.      ,   0.      ,\n          0.      ],\n       [       nan,        nan,        nan, ...,   0.      ,   0.      ,\n          0.      ],\n       ...,\n       [  0.      ,   0.      ,   0.      , ...,   0.      ,   0.      ,\n          0.      ],\n       [       nan,        nan,        nan, ...,   0.      ,   0.      ,\n          0.      ],\n       [  0.      ,   0.      ,   0.      , ...,   0.      ,   0.      ,\n          0.      ]], shape=(2048, 2048), dtype=float32), array([[  46.4375   ,  -21.75     ,  -36.71875  , ...,  -39.96875  ,\n           3.125    , -147.25     ],\n       [ -13.859375 ,  -17.453125 ,   17.375    , ...,   57.90625  ,\n          25.859375 ,  -68.0625   ],\n       [  34.1875   ,   -9.421875 ,  -49.       , ...,  -46.71875  ,\n         -64.5625   ,   37.6875   ],\n       ...,\n       [   3.3261719,  -60.03125  ,  -64.       , ...,   18.328125 ,\n         -17.234375 ,  -30.90625  ],\n       [  -0.9145508,   97.25     ,  -18.109375 , ...,   35.75     ,\n         -45.0625   ,   21.78125  ],\n       [ -36.5      ,  -63.8125   ,   68.125    , ...,  -48.09375  ,\n         -29.515625 ,   11.1640625]], shape=(2048, 2048), dtype=float32), rtol=0.02, atol=0.02)
E        +    where <function allclose at 0x706316368d70> = np.allclose

task.py:262: AssertionError
=========================== short test summary info ============================
FAILED task.py::TestMatmulOCL::test_correctness_wrt_pytorch[0] - AssertionErr...
FAILED task.py::TestMatmulOCL::test_correctness_wrt_pytorch[1] - AssertionErr...
FAILED task.py::TestMatmulOCL::test_correctness_wrt_reference[0] - AssertionE...
FAILED task.py::TestMatmulOCL::test_correctness_wrt_reference[1] - AssertionE...
======================= 4 failed, 1 deselected in 0.89s ========================

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
- **Online Algorithms**: Use single-pass algorithms with running statistics. For softmax: maintain running_max and running_sum, rescaling sum when max changes. For variance: use Welford's online algorithm.
- **Work-Group Reductions**: Replace atomic operations with O(log N) tree-based reductions in local memory. Synchronize with `group_barrier()` between iterations.