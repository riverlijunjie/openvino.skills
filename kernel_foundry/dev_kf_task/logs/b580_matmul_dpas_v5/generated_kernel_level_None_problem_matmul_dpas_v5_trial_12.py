1. **Analysis**

### What was good / bad in previous kernels
- **Correctness is fine** in all tested versions.
- Performance stayed flat (~33.9 ms) because the DPAS usage was not mapped efficiently:
  - Previous kernels often did **1 output per thread** (`float acc`), which underuses XMX/DPAS throughput.
  - `B` fragment assembly used **16 scalar SLM loads** into `half16` every DPAS call (`for kk ... b_vec.s[kk] = ...`), which is costly.
  - Two barriers per K-step with limited compute between barriers hurt latency hiding.
- Version-1 attempted bigger tile output per subgroup, but mapping was awkward and still had scalar gathers.

### Key bottlenecks
1. **Low arithmetic intensity per thread** (single accumulator).
2. **Non-vector-friendly B layout** causing scalar gathers.
3. **Barrier-heavy loop structure** with insufficient compute between sync points.
4. Under-leveraged subgroup-level reuse.

### Proposed optimizations
- Keep DPAS (`intel_sub_group_f16_f16_matrix_mad_k16`) as required.
- Use **register blocking**: each work-item computes multiple outputs (THREAD_M×THREAD_N style).  
  Here: per lane compute 2 rows × 1 col (two accumulators), improving flops/byte.
- Use subgroup-friendly mapping:
  - WG = `16x8`, two subgroups (each subgroup handles 8 rows × 16 cols tile).
  - A/B tiles staged in SLM once per K-block.
- Keep K blocking at 16 to match DPAS k16.
- Tail K handled scalar for exact semantics.
- Boundary-safe loads/stores.

> Note: GEMM has no cross-thread accumulation for a C element in this mapping, so **work-group reduction is not applicable** (there are no atomics to replace). Forcing an artificial reduction would degrade performance and complicate correctness.

---

2. **Improved OCL code**
```OCL
// FP16 GEMM: C[M,N] = A[M,K] x B[K,N]
// Input/Output: half, accumulation: float (reference-equivalent)
//
// Launch metadata (host-side):
//   LWS = {16, 8, 1}                         // 128 threads/WG, 2 subgroups
//   GWS = {round_up(N,16), round_up(M,16),1} // 1 WG -> 16x16 C tile
//   Subgroup size = 16
//
// Mapping:
//   - WG computes one 16x16 output tile
//   - 2 subgroups along local_y: each subgroup computes 8x16 rows
//   - Per lane (column): register block THREAD_M=2, THREAD_N=1
//     => each lane accumulates 2 rows for one column
//
// DPAS:
//   - Uses intel_sub_group_f16_f16_matrix_mad_k16 on k16 chunks
//   - Accumulators are float8 (DPAS native form), then extracted

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_matrix_multiply_accumulate : enable

#define TILE_M 16
#define TILE_N 16
#define TILE_K 16

#define SG_SIZE 16
#define WG_Y 8

__attribute__((reqd_work_group_size(SG_SIZE, WG_Y, 1)))
__attribute__((intel_reqd_sub_group_size(SG_SIZE)))
__kernel void matmul(
    __global const half* restrict A,
    __global const half* restrict B,
    __global half* restrict C,
    const int M,
    const int K,
    const int N)
{
    const int lx = get_local_id(0);      // 0..15 (lane/column in tile)
    const int ly = get_local_id(1);      // 0..7  (row within subgroup tile)
    const int sg = get_sub_group_id();   // 0..1

    const int wg_x = get_group_id(0);
    const int wg_y = get_group_id(1);

    const int tile_col0 = wg_x * TILE_N;
    const int tile_row0 = wg_y * TILE_M;

    // subgroup row-base inside 16x16 tile: sg0->rows[0..7], sg1->rows[8..15]
    const int row_base = tile_row0 + sg * 8 + ly;
    const int col      = tile_col0 + lx;

    // Register blocking: 2 rows x 1 col per work-item
    float acc0 = 0.0f;
    float acc1 = 0.0f; // second row offset +4 within each 8-row subgroup half

    // SLM tiles (+1 pad to ease bank conflicts)
    __local half Asub[TILE_M][TILE_K + 1];
    __local half Bsub[TILE_K][TILE_N + 1];

    const int k_full_end = (K / TILE_K) * TILE_K;

    for (int kb = 0; kb < k_full_end; kb += TILE_K) {
        // Cooperative load: each thread loads 2 A elems + 2 B elems
        int t = ly * SG_SIZE + lx; // 0..127

        // Load A tile 16x16 (256 elems)
        #pragma unroll
        for (int it = 0; it < 2; ++it) {
            int idx = t + it * 128;   // 0..255
            int ar = idx >> 4;        // /16
            int ak = idx & 15;        // %16
            int gr = tile_row0 + ar;
            int gk = kb + ak;
            Asub[ar][ak] = (gr < M && gk < K) ? A[gr * K + gk] : (half)0.0h;
        }

        // Load B tile 16x16 (256 elems)
        #pragma unroll
        for (int it = 0; it < 2; ++it) {
            int idx = t + it * 128;   // 0..255
            int bk = idx >> 4;        // /16
            int bc = idx & 15;        // %16
            int gk = kb + bk;
            int gc = tile_col0 + bc;
            Bsub[bk][bc] = (gk < K && gc < N) ? B[gk * N + gc] : (half)0.0h;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Build B vector for this lane/column (k16)
        half16 b_vec;
        #pragma unroll
        for (int kk = 0; kk < 16; ++kk) {
            b_vec.s[kk] = Bsub[kk][lx];
        }

        // Row 0 in register block
        {
            int r0 = sg * 8 + ly;
            half16 a_vec0 = vload16(0, &Asub[r0][0]);
            float8 z0 = (float8)(0.0f);
            z0 = intel_sub_group_f16_f16_matrix_mad_k16(a_vec0, b_vec, z0);
            acc0 += z0.s0;
        }

        // Row 1 in register block (offset +4 within subgroup half: improves reuse / ILP)
        {
            int r1 = sg * 8 + ((ly + 4) & 7);
            half16 a_vec1 = vload16(0, &Asub[r1][0]);
            float8 z1 = (float8)(0.0f);
            z1 = intel_sub_group_f16_f16_matrix_mad_k16(a_vec1, b_vec, z1);
            acc1 += z1.s0;
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Tail K + store
    if (col < N) {
        int r0g = row_base;
        int r1g = tile_row0 + sg * 8 + ((ly + 4) & 7);

        if (r0g < M) {
            float v0 = acc0;
            for (int k = k_full_end; k < K; ++k) {
                v0 += convert_float(A[r0g * K + k]) * convert_float(B[k * N + col]);
            }
            C[r0g * N + col] = convert_half(v0);
        }

        if (r1g < M) {
            float v1 = acc1;
            for (int k = k_full_end; k < K; ++k) {
                v1 += convert_float(A[r1g * K + k]) * convert_float(B[k * N + col]);
            }
            C[r1g * N + col] = convert_half(v1);
        }
    }
}
```

If you want, I can provide a second variant with `LWS={16,16,1}` and 4 accumulators/thread (higher register blocking) to A/B test on Battlemage.

