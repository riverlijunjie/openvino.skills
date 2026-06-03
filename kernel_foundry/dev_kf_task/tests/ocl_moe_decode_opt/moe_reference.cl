// [REFERENCE_START]
// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Standalone mlp_gate_up kernel extracted from moe_3gemm_swiglu_mlp.cl
// Parameters hardcoded for PTL 12Xe (Xe3) with the following model config:
//   - 128 experts, top-8, hidden=2048, intermediate=768
//   - u4 weights, f16 scales, u4 zero points, group_size=128
//   - SwiGLU activation

// ============= JIT Macros (hardcoded for PTL config) =============
#define GATE_UP_ENABLE       1
#define SHARED_EXPERT_ENABLE 0
#define SUBGROUP_SIZE        32
#define SUBGROUP_NUM         8
#define MAX_TOPK             8
#define EXPERT_NUM           128
#define HIDDEN_SIZE          2048
#define INTERMEDIATE_SIZE    768
#define N_BLOCK              4
#define GATE_UP_GROUP_SIZE   128
#define DOWN_GROUP_SIZE      128
#define HAS_ZP               1
#define WEIGHT_IS_SIGNED     0
#define WEIGHT_COMPRESSEION_DT 0
#define MOE_WEI_DT           uchar
#define MOE_SCALE_DT         half
#define MOE_ZP_DT            uchar
#define MOE_DTYPE            half
#define MOE_DTYPE_SIZE       2

// ============= Derived Macros =============
#define unroll_for __attribute__((opencl_unroll_hint)) for

// FAKE_GROUP_SIZE: min(GATE_UP_GROUP_SIZE, DOWN_GROUP_SIZE) clamped to 128
#if GATE_UP_GROUP_SIZE < DOWN_GROUP_SIZE
#    define MOE_MIN_GROUP_SIZE GATE_UP_GROUP_SIZE
#else
#    define MOE_MIN_GROUP_SIZE DOWN_GROUP_SIZE
#endif
#if MOE_MIN_GROUP_SIZE < 128
#    define FAKE_GROUP_SIZE MOE_MIN_GROUP_SIZE
#else
#    define FAKE_GROUP_SIZE 128
#endif

// Number of K-elements each work-item handles per gk-iteration
#define ELEMS_PER_LANE (FAKE_GROUP_SIZE / SUBGROUP_SIZE)

// Experts per token
#if SHARED_EXPERT_ENABLE
#define EXPERTS_PER_TOKEN (MAX_TOPK + 1)
#else
#define EXPERTS_PER_TOKEN MAX_TOPK
#endif

// Gate activation: SwiGLU (Swish)
#define MOE_GATE_ACT(x) ((x) / (1.0f + exp(-(x))))

// HAS_ZP: asymmetric quantization (subtract zero point)
#define ZP_ADJUST_2(sum0, sum1, xg_sum_gk, z0, z1) ((sum0) - (xg_sum_gk) * (z0)), ((sum1) - (xg_sum_gk) * (z1))
#define ZP_ADJUST_4(sum0123, xg_sum_gk, z) ((sum0123) - (xg_sum_gk) * (z))

// WEIGHT_IS_SIGNED = 0: unsigned u4
#define DEQUANT_4BIT_LO(v) convert_half((v) & 0x0F)
#define DEQUANT_4BIT_HI(v) convert_half((v) >> 4)

// ============= gate_up_gemv_n2x_u4 inline function =============
inline void gate_up_gemv_n2x_u4(const __global uchar* weight,
                                const __global half* scales,
                                const __global uchar* zps,
                                __global half* y,
                                int N,
                                int K,
                                __local half* x2,
                                __local float* xg_sum,
                                const bool silu) {
    int num_sg = get_num_sub_groups();
    int id_local = get_sub_group_local_id();

    int n_start = get_global_id(2) * N_BLOCK;
    int n_end = n_start + N_BLOCK;
    unroll_for(int n = n_start; n < n_end; n += 2) {
        const __global uchar* B = weight + n * K / 2;
        float sum_all0 = 0;
        float sum_all1 = 0;
        const __global half* S = scales + n;
        const __global uchar* Z = zps + n / 2;

        unroll_for(int gk = 0; gk < K / FAKE_GROUP_SIZE; gk++) {
            int scale_offset = (gk * FAKE_GROUP_SIZE / GATE_UP_GROUP_SIZE) * N;
            half s0 = S[scale_offset];
            half s1 = S[scale_offset + 1];

            int zp_offset = (gk * FAKE_GROUP_SIZE / GATE_UP_GROUP_SIZE) * N / 2;
            uchar z = Z[zp_offset];
            half z_hf0 = convert_half(z & 0xf);
            half z_hf1 = convert_half(z >> 4);

#if ELEMS_PER_LANE == 4
            half2 sum0;
            half2 sum1;
            half4 a = as_half4(intel_sub_group_block_read_us4((const __local ushort*)x2 + gk * FAKE_GROUP_SIZE));
            uchar2 b = intel_sub_group_block_read_uc2((const __global uchar*)B + gk * FAKE_GROUP_SIZE / 2);
            uchar2 b2 = intel_sub_group_block_read_uc2((const __global uchar*)(B + (K / 2) + gk * FAKE_GROUP_SIZE / 2));

            sum0.s0 = fma(a.s0, (DEQUANT_4BIT_LO(b.s0)), 0);
            sum0.s1 = fma(a.s1, (DEQUANT_4BIT_LO(b.s1)), 0);
            sum0.s0 = fma(a.s2, (DEQUANT_4BIT_HI(b.s0)), sum0.s0);
            sum0.s1 = fma(a.s3, (DEQUANT_4BIT_HI(b.s1)), sum0.s1);

            sum1.s0 = fma(a.s0, (DEQUANT_4BIT_LO(b2.s0)), 0);
            sum1.s1 = fma(a.s1, (DEQUANT_4BIT_LO(b2.s1)), 0);
            sum1.s0 = fma(a.s2, (DEQUANT_4BIT_HI(b2.s0)), sum1.s0);
            sum1.s1 = fma(a.s3, (DEQUANT_4BIT_HI(b2.s1)), sum1.s1);

            sum_all0 += (sum0[0] + sum0[1] - xg_sum[gk] * z_hf0) * s0;
            sum_all1 += (sum1[0] + sum1[1] - xg_sum[gk] * z_hf1) * s1;
#elif ELEMS_PER_LANE == 2
            half sum0;
            half sum1;
            half2 a = as_half2(intel_sub_group_block_read_us2((const __local ushort*)x2 + gk * FAKE_GROUP_SIZE));
            uchar b = intel_sub_group_block_read_uc((const __global uchar*)B + gk * FAKE_GROUP_SIZE / 2);
            uchar b2 = intel_sub_group_block_read_uc((const __global uchar*)(B + (K / 2) + gk * FAKE_GROUP_SIZE / 2));

            sum0 = fma(a.s0, (DEQUANT_4BIT_LO(b)), (half)0);
            sum0 = fma(a.s1, (DEQUANT_4BIT_HI(b)), sum0);

            sum1 = fma(a.s0, (DEQUANT_4BIT_LO(b2)), (half)0);
            sum1 = fma(a.s1, (DEQUANT_4BIT_HI(b2)), sum1);

            sum_all0 += (sum0 - xg_sum[gk] * z_hf0) * s0;
            sum_all1 += (sum1 - xg_sum[gk] * z_hf1) * s1;
#else
            half4 sum0;
            half4 sum1;
            half8 a = as_half8(intel_sub_group_block_read_us8((const __local ushort*)x2 + gk * FAKE_GROUP_SIZE));
            uchar4 b = intel_sub_group_block_read_uc4((const __global uchar*)B + gk * FAKE_GROUP_SIZE / 2);
            uchar4 b2 = intel_sub_group_block_read_uc4((const __global uchar*)(B + (K / 2) + gk * FAKE_GROUP_SIZE / 2));

            sum0.s0 = fma(a.s0, (DEQUANT_4BIT_LO(b.s0)), 0);
            sum0.s1 = fma(a.s1, (DEQUANT_4BIT_LO(b.s1)), 0);
            sum0.s2 = fma(a.s2, (DEQUANT_4BIT_LO(b.s2)), 0);
            sum0.s3 = fma(a.s3, (DEQUANT_4BIT_LO(b.s3)), 0);

            sum0.s0 = fma(a.s4, (DEQUANT_4BIT_HI(b.s0)), sum0.s0);
            sum0.s1 = fma(a.s5, (DEQUANT_4BIT_HI(b.s1)), sum0.s1);
            sum0.s2 = fma(a.s6, (DEQUANT_4BIT_HI(b.s2)), sum0.s2);
            sum0.s3 = fma(a.s7, (DEQUANT_4BIT_HI(b.s3)), sum0.s3);

            sum1.s0 = fma(a.s0, (DEQUANT_4BIT_LO(b2.s0)), 0);
            sum1.s1 = fma(a.s1, (DEQUANT_4BIT_LO(b2.s1)), 0);
            sum1.s2 = fma(a.s2, (DEQUANT_4BIT_LO(b2.s2)), 0);
            sum1.s3 = fma(a.s3, (DEQUANT_4BIT_LO(b2.s3)), 0);

            sum1.s0 = fma(a.s4, (DEQUANT_4BIT_HI(b2.s0)), sum1.s0);
            sum1.s1 = fma(a.s5, (DEQUANT_4BIT_HI(b2.s1)), sum1.s1);
            sum1.s2 = fma(a.s6, (DEQUANT_4BIT_HI(b2.s2)), sum1.s2);
            sum1.s3 = fma(a.s7, (DEQUANT_4BIT_HI(b2.s3)), sum1.s3);

            sum_all0 += (sum0[0] + sum0[1] + sum0[2] + sum0[3] - xg_sum[gk] * z_hf0) * s0;
            sum_all1 += (sum1[0] + sum1[1] + sum1[2] + sum1[3] - xg_sum[gk] * z_hf1) * s1;
#endif
        }

        sum_all0 = sub_group_reduce_add(sum_all0);
        sum_all1 = sub_group_reduce_add(sum_all1);
        if (id_local == 0) {
            if (silu) {
                y[n] *= MOE_GATE_ACT(sum_all0);
                y[n + 1] *= MOE_GATE_ACT(sum_all1);
            } else {
                y[n] = sum_all0;
                y[n + 1] = sum_all1;
            }
        }
    }
}

// ============= Main Kernel =============
__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE)))
__kernel void mlp_gate_up(
    const __global int* expert_list,       // [1 x MAX_TOPK] = [1x8]
    const __global uchar* gate_weight_addr,  // [128 x 768 x 2048] in u4 packed
    const __global half* gate_scale_addr,    // [128 x 768 x 16]
    const __global uchar* gate_zp_addr,      // [128 x 768 x 16] in u4 packed
    const __global uchar* up_weight_addr,    // [128 x 768 x 2048] in u4 packed
    const __global half* up_scale_addr,      // [128 x 768 x 16]
    const __global uchar* up_zp_addr,        // [128 x 768 x 16] in u4 packed
    __global half* x,                        // [1 x 2048]
    __global half* y)                        // [8 x 768]
{
    // global: [token_num*EXPERTS_PER_TOKEN, SUBGROUP_SIZE, N//N_BLOCK] = [8, 32, 192]
    // local:  [1, SUBGROUP_SIZE, SUBGROUP_NUM] = [1, 32, 8]
    int flat_id = get_global_id(0);
    int token_idx = flat_id / EXPERTS_PER_TOKEN;
    int expert_slot = flat_id % EXPERTS_PER_TOKEN;
    y += flat_id * INTERMEDIATE_SIZE;

    // Weight sizes for u4: N * K / 2 bytes
    const int expert_wei_size = INTERMEDIATE_SIZE * HIDDEN_SIZE / 2;
    const int expert_scale_size = INTERMEDIATE_SIZE * HIDDEN_SIZE / GATE_UP_GROUP_SIZE;
    const int expert_zp_size = INTERMEDIATE_SIZE * HIDDEN_SIZE / 2 / GATE_UP_GROUP_SIZE;

    int expert_id = expert_list[token_idx * MAX_TOPK + expert_slot];

    // gate, [HIDDEN_SIZE, INTERMEDIATE_SIZE] stored as [N, K] in u4
    __global uchar* gate_weight = (__global uchar*)(gate_weight_addr + expert_id * expert_wei_size);
    __global half* gate_scale = (__global half*)(gate_scale_addr + expert_id * expert_scale_size);
    __global uchar* gate_zp = (__global uchar*)(gate_zp_addr + expert_id * expert_zp_size);

    // up, [HIDDEN_SIZE, INTERMEDIATE_SIZE] stored as [N, K] in u4
    __global uchar* up_weight = (__global uchar*)(up_weight_addr + expert_id * expert_wei_size);
    __global half* up_scale = (__global half*)(up_scale_addr + expert_id * expert_scale_size);
    __global uchar* up_zp = (__global uchar*)(up_zp_addr + expert_id * expert_zp_size);

    #if GATE_UP_GROUP_SIZE % FAKE_GROUP_SIZE != 0
    if (get_sub_group_id() == 0 && get_sub_group_local_id() == 0) {
        printf("GATE_UP_GROUP_SIZE(%d) must be a multiple of FAKE_GROUP_SIZE(%d)", GATE_UP_GROUP_SIZE, FAKE_GROUP_SIZE);
    }
    return;
    #endif

    __local half x2[HIDDEN_SIZE];
    __local float xg_sum[HIDDEN_SIZE / FAKE_GROUP_SIZE];

    // Interleaving x into x2 for u4 path
    int id_sg = get_sub_group_id();
    int num_sg = get_num_sub_groups();
    int id_local = get_sub_group_local_id();
    __global half* px = x + token_idx * HIDDEN_SIZE + id_sg * FAKE_GROUP_SIZE;
    __local half* px2 = x2 + id_sg * FAKE_GROUP_SIZE;
    unroll_for(int i = id_sg; i < HIDDEN_SIZE / FAKE_GROUP_SIZE; i += num_sg, px += num_sg * FAKE_GROUP_SIZE, px2 += num_sg * FAKE_GROUP_SIZE) {
        float x_group_sum = 0;
        unroll_for(int j = id_local; j < FAKE_GROUP_SIZE / 2; j += SUBGROUP_SIZE) {
            half even = px[2 * j + 0];
            half odd = px[2 * j + 1];
            px2[j] = even;
            px2[j + FAKE_GROUP_SIZE / 2] = odd;
            x_group_sum += even + odd;
        }
        x_group_sum = sub_group_reduce_add(x_group_sum);
        if (id_local == 0) {
            xg_sum[i] = x_group_sum / SUBGROUP_SIZE;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // First compute up projection (no activation), then gate projection (with SwiGLU)
    gate_up_gemv_n2x_u4(up_weight, up_scale, up_zp, y, INTERMEDIATE_SIZE, HIDDEN_SIZE, x2, xg_sum, false);
    gate_up_gemv_n2x_u4(gate_weight, gate_scale, gate_zp, y, INTERMEDIATE_SIZE, HIDDEN_SIZE, x2, xg_sum, true);
}

// [REFERENCE_END]
