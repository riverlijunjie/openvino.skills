1. **Analysis**

- The prior kernels are correct, but performance is flat (~33.9 ms) mainly because DPAS is not mapped to its natural fragment ownership:
  - They call `intel_sub_group_f16_f16_matrix_mad_k16(...)` but only consume `s0`, wasting most computed outputs.
  - B operand assembly is scalar-gather heavy (`for kk ... b_vec.s[kk] = ...`), which inflates instruction count and SLM traffic.
  - Two barriers per K-slice without real overlap means memory latency is poorly hidden.
- Version-to-version differences didn’t improve runtime because both kept the same fundamental inefficiency: scalar-output mindset around a matrix instruction.
- Also, “kernel fusion” is not applicable here functionally (this kernel is a single GEMM op). We must preserve exact `C = A x B` semantics and cannot fuse unrelated ops.

### Proposed changes

- **Subgroup-native mapping**: one subgroup computes a **16x16 C tile**, with each lane owning one output column and 16 row accumulators split as `float8 + float8`.  
  This uses all DPAS-produced components, addressing the largest waste.
- **SLM tiling**:
  - `Asub[16][16+1]`, `Bsub[16][16+1]` (padding reduces bank conflicts).
  - Cooperative load by 16 lanes.
- **DPAS-friendly operand construction**:
  - `a_vec` loaded via `vload16` from row-contiguous A tile.
  - `b_vec` assembled once per lane per K-block from column of B tile (still explicit due intrinsic operand shape).
- **Lower sync overhead vs 16x16 WG design**:
  - Use `WG = 16x1` (one subgroup per WG) so barriers are cheap and occupancy is high.
- **Exact boundary/tail handling**:
  - Full-tile DPAS loop for `K/16`.
  - Scalar cleanup for `K` tail.
  - M/N edge checks on store and tail path.
- **Collectives note**:
  - For this DPAS mapping, subgroup collectives are not the main win, but subgroup execution is explicitly fixed (`intel_reqd_sub_group_size(16)`), and mapping is subgroup-friendly.

---

2. **Improved OCL code**

```OCL
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_matrix_multiply_accumulate : enable

// -----------------------------------------------------------------------------
// Intel Xe2 (Battlemage) DPAS-oriented FP16 GEMM
//   C[M,N] = A[M,K] x B[K,N]
//   input/output: half, accumulation: float
//
// Suggested launch metadata (host side):
//   LWS = {16, 1, 1}                         // exactly 1 subgroup / WG
//   GWS = { round_up(N,16), ceil_div(M,16), 1 }
//   subgroup size hint = 16
//
// Mapping:
//   - One WG (one subgroup) computes a 16x16 C tile:
//       rows: tile_row ... tile_row+15
//       cols: tile_col ... tile_col+15
//   - lane id selects column within the 16-column tile
//   - each lane accumulates 16 rows for its column (float8 lo + float8 hi)
//   - K processed in chunks of 16 using DPAS k16 intrinsic
// -----------------------------------------------------------------------------

#define TM 16
#define TN 16
#define TK 16

__attribute__((reqd_work_group_size(16,1,1)))
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void matmul(
    __global const half* restrict A,
    __global const half* restrict B,
    __global half* restrict C,
    const int M,
    const int K,
    const int N)
{
    const int lane = get_local_id(0);   // 0..15
    const int gx   = get_group_id(0);
    const int gy   = get_group_id(1);

    const int tile_row = gy * TM;
    const int tile_col = gx * TN;
    const int col      = tile_col + lane;

    // SLM tiles (+1 padding to reduce bank conflicts)
    __local half Asub[TM][TK + 1];
    __local half Bsub[TK][TN + 1];

    float8 acc_lo = (float8)(0.0f); // rows 0..7  for this lane's column
    float8 acc_hi = (float8)(0.0f); // rows 8..15 for this lane's column

    const int k_full_end = (K / TK) * TK;

    for (int kb = 0; kb < k_full_end; kb += TK) {
        // Cooperative load A tile: each lane loads one K-column for all 16 rows
        #pragma unroll
        for (int r = 0; r < TM; ++r) {
            const int gr = tile_row + r;
            const int gk = kb + lane;
            Asub[r][lane] = (gr < M && gk < K) ? A[gr * K + gk] : (half)0.0h;
        }

        // Cooperative load B tile: each lane loads one N-column for all 16 K rows
        #pragma unroll
        for (int kk = 0; kk < TK; ++kk) {
            const int gk = kb + kk;
            const int gc = col;
            Bsub[kk][lane] = (gk < K && gc < N) ? B[gk * N + gc] : (half)0.0h;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Build B vector for this lane's output column: B[kb:kb+15, col]
        half16 b_vec;
        #pragma unroll
        for (int kk = 0; kk < 16; ++kk) {
            b_vec.s[kk] = Bsub[kk][lane];
        }

        // Rows 0..7
        {
            // Subgroup-consistent A fragment selection
            const int src_row = (lane < 8) ? lane : 0;
            half16 a_vec = vload16(0, &Asub[src_row][0]);
            acc_lo = intel_sub_group_f16_f16_matrix_mad_k16(a_vec, b_vec, acc_lo);
        }

        // Rows 8..15
        {
            const int src_row = (lane < 8) ? (lane + 8) : 8;
            half16 a_vec = vload16(0, &Asub[src_row][0]);
            acc_hi = intel_sub_group_f16_f16_matrix_mad_k16(a_vec, b_vec, acc_hi);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store with exact tail-K accumulation
    if (col < N) {
        float tmp[8];

        // lo rows 0..7
        tmp[0] = acc_lo.s0; tmp[1] = acc_lo.s1; tmp[2] = acc_lo.s2; tmp[3] = acc_lo.s3;
        tmp[4] = acc_lo.s4; tmp[5] = acc_lo.s5; tmp[6] = acc_lo.s6; tmp[7] = acc_lo.s7;

        #pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int gr = tile_row + r;
            if (gr < M) {
                float v = tmp[r];
                for (int k = k_full_end; k < K; ++k) {
                    v += convert_float(A[gr * K + k]) * convert_float(B[k * N + col]);
                }
                C[gr * N + col] = convert_half(v);
            }
        }

        // hi rows 8..15
        tmp[0] = acc_hi.s0; tmp[1] = acc_hi.s1; tmp[2] = acc_hi.s2; tmp[3] = acc_hi.s3;
        tmp[4] = acc_hi.s4; tmp[5] = acc_hi.s5; tmp[6] = acc_hi.s6; tmp[7] = acc_hi.s7;

        #pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int gr = tile_row + 8 + r;
            if (gr < M) {
                float v = tmp[r];
                for (int k = k_full_end; k < K; ++k) {
                    v += convert_float(A[gr * K + k]) * convert_float(B[k * N + col]);
                }
                C[gr * N + col] = convert_half(v);
            }
        }
    }
}
```

If you want, I can also provide a second variant with **SLM ping-pong double buffering** (more complex, but often faster on Battlemage when K is large).

