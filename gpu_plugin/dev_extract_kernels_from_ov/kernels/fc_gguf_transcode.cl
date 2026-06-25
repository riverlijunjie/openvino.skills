// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// GGUF weight transcode kernel (compute-bound / large-M prefill path).
//
// Converts a raw GGUF block-quantised weight matrix W[N, K] into a OneDNN-WOQ-native low-bit layout:
//   - a packed weight scratchpad: i4 (TRANSCODE_TO_I4=1) or i8 (signed), in [N, K] physical order
//     (matches dnnl wei_md [K,N] with format_tag::ba), and
//   - a parallel f16 per-group scale scratchpad [K/REQUANT_GROUP, N] = dnnl scale md [K/group, N]
//     with element (g, n) at g*N + n (per-K-group x per-N mask).
//
// The block bytes are decoded to half in registers with the SAME per-format decoders used by the
// native GEMV kernel (so numerics track exactly), then symmetrically re-quantized per REQUANT_GROUP
// elements to the target low-bit domain. dequant NEVER lands in an f16/f32 weight buffer (constraint
// C2): the only persisted weight is the low-bit scratchpad; the f16 values live only in registers.
//
// One work-item owns one (n, GGUF block): global = [N_SIZE, K_SIZE / GGUF_BLOCK_ELEM, 1], local = [SG, 1, 1].
// The block is decoded once and every REQUANT group inside it is requantized from the shared decoded
// window, so the heavy bit-unpacking runs a single time per block instead of
// (GGUF_BLOCK_ELEM / REQUANT_GROUP)x (8x for K-quants with a 256-elem block and a 32-elem group).
// REQUANT_GROUP must divide GGUF_BLOCK_ELEM (so a group never straddles two GGUF blocks).

#include "include/batch_headers/common.cl"
#include "gguf/gguf_iq_tables.hpp"

inline half FUNC(tq_load_f16)(const __global uchar* p) {
    ushort bits = (ushort)p[0] | ((ushort)p[1] << 8);
    return as_half(bits);
}

// ---- per-format block decoders (identical math to fc_gguf_opt.cl) ----

#if defined(GGUF_IS_Q4_0)
inline void FUNC(tq_decode_block)(const __global uchar* blk, __private half* out) {
    const half d = FUNC_CALL(tq_load_f16)(blk);
    const __global uchar* qs = blk + 2;
    for (int j = 0; j < 16; ++j) {
        out[j]      = (half)(((int)(qs[j] & 0x0F) - 8) * (float)d);
        out[j + 16] = (half)(((int)(qs[j] >> 4)   - 8) * (float)d);
    }
}
#endif

#if defined(GGUF_IS_Q8_0)
inline void FUNC(tq_decode_block)(const __global uchar* blk, __private half* out) {
    const half d = FUNC_CALL(tq_load_f16)(blk);
    const __global char* qs = (const __global char*)(blk + 2);
    for (int j = 0; j < 32; ++j) {
        out[j] = (half)((float)qs[j] * (float)d);
    }
}
#endif

#if defined(GGUF_IS_Q4_K) || defined(GGUF_IS_Q5_K)
inline void FUNC(tq_scale_min_k4)(int j, const __global uchar* q, uchar* d, uchar* m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (uchar)((q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4));
        *m = (uchar)((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
    }
}
#endif

#if defined(GGUF_IS_Q4_K)
inline void FUNC(tq_decode_block)(const __global uchar* blk, __private half* out) {
    const float d    = (float)FUNC_CALL(tq_load_f16)(blk);
    const float dmin = (float)FUNC_CALL(tq_load_f16)(blk + 2);
    const __global uchar* scales = blk + 4;
    const __global uchar* qs     = blk + 16;
    int o = 0, is = 0;
    for (int j = 0; j < 256; j += 64) {
        uchar sc, m;
        FUNC_CALL(tq_scale_min_k4)(is + 0, scales, &sc, &m);
        const float d1 = d * sc, m1 = dmin * m;
        FUNC_CALL(tq_scale_min_k4)(is + 1, scales, &sc, &m);
        const float d2 = d * sc, m2 = dmin * m;
        for (int l = 0; l < 32; ++l) out[o++] = (half)(d1 * (float)(qs[l] & 0x0F) - m1);
        for (int l = 0; l < 32; ++l) out[o++] = (half)(d2 * (float)(qs[l] >> 4) - m2);
        qs += 32; is += 2;
    }
}
#endif

#if defined(GGUF_IS_Q5_K)
inline void FUNC(tq_decode_block)(const __global uchar* blk, __private half* out) {
    const float d    = (float)FUNC_CALL(tq_load_f16)(blk);
    const float dmin = (float)FUNC_CALL(tq_load_f16)(blk + 2);
    const __global uchar* scales = blk + 4;
    const __global uchar* qh     = blk + 16;
    const __global uchar* ql     = blk + 48;
    int o = 0, is = 0; uchar u1 = 1, u2 = 2;
    for (int j = 0; j < 256; j += 64) {
        uchar sc, m;
        FUNC_CALL(tq_scale_min_k4)(is + 0, scales, &sc, &m);
        const float d1 = d * sc, m1 = dmin * m;
        FUNC_CALL(tq_scale_min_k4)(is + 1, scales, &sc, &m);
        const float d2 = d * sc, m2 = dmin * m;
        for (int l = 0; l < 32; ++l) { const int q = (int)(ql[l] & 0x0F) + ((qh[l] & u1) ? 16 : 0); out[o++] = (half)(d1 * (float)q - m1); }
        for (int l = 0; l < 32; ++l) { const int q = (int)(ql[l] >> 4)   + ((qh[l] & u2) ? 16 : 0); out[o++] = (half)(d2 * (float)q - m2); }
        ql += 32; is += 2; u1 <<= 2; u2 <<= 2;
    }
}
#endif

#if defined(GGUF_IS_Q6_K)
inline void FUNC(tq_decode_block)(const __global uchar* blk, __private half* out) {
    const __global uchar* ql = blk;
    const __global uchar* qh = blk + 128;
    const __global char*  sc = (const __global char*)(blk + 192);
    const float d = (float)FUNC_CALL(tq_load_f16)(blk + 208);
    int o = 0;
    for (int n = 0; n < 256; n += 128) {
        for (int l = 0; l < 32; ++l) {
            const int is = l / 16;
            const int q1 = (int)((ql[l + 0]  & 0x0F) | (((qh[l] >> 0) & 3) << 4)) - 32;
            const int q2 = (int)((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32;
            const int q3 = (int)((ql[l + 0]  >> 4)   | (((qh[l] >> 4) & 3) << 4)) - 32;
            const int q4 = (int)((ql[l + 32] >> 4)   | (((qh[l] >> 6) & 3) << 4)) - 32;
            out[o + l + 0]  = (half)(d * (float)sc[is + 0] * q1);
            out[o + l + 32] = (half)(d * (float)sc[is + 2] * q2);
            out[o + l + 64] = (half)(d * (float)sc[is + 4] * q3);
            out[o + l + 96] = (half)(d * (float)sc[is + 6] * q4);
        }
        o += 128; ql += 64; qh += 32; sc += 8;
    }
}
#endif

#if defined(GGUF_IS_Q3_K)
// Q3_K block layout (110 bytes / 256 elements; ggml block_q3_k). Same unpack as
// frontend dequantize_q3_k() and fc_gguf_opt.cl GGUF_IS_Q3_K: 2-bit payload plus
// high-mask correction -> q in {-4..3}, multiplied by per-16 signed sub-scale and d_all.
//
// Optimisations vs. the literal CPU port (numerically bit-exact: only the SCHEDULE changes,
// every individual FP op uses the same operands in the same order as the naive version):
//   * 32B hmask + 64B qs are pulled into private memory ONCE. The naive loop re-reads each
//     qs byte 4x (one per j shift) and each hmask byte 8x (4 shifts x 2 halves) from
//     __global, i.e. ~384 redundant uchar loads per block.
//   * The 12-byte packed scales are decoded into sixteen pre-scaled sub-scales (dl[0..15])
//     up front, so the inner lane loop is a flat dl[] lookup instead of a 4-way ternary on
//     `is` plus a byte-shift on a uint.
//   * Both the n (half) and j (shift) outer loops are hinted to unroll, so shift/mask/dl-index
//     are compile-time constants -> the 16-lane lane loop SIMD-vectorises cleanly.
inline void FUNC(tq_decode_block)(const __global uchar* blk, __private half* out) {
    // 1. Cache global block bytes once.
    __private uchar hmask[32];
    __private uchar qs[64];
    __attribute__((opencl_unroll_hint))
    for (int i = 0; i < 32; ++i) hmask[i] = blk[i];           // hmask = blk[0..31]
    __attribute__((opencl_unroll_hint))
    for (int i = 0; i < 64; ++i) qs[i]    = blk[32 + i];      // qs    = blk[32..95]

    const float d_all = (float)FUNC_CALL(tq_load_f16)(blk + 108);

    // 2. Decode the 12-byte packed scales -> sixteen pre-scaled sub-scales.
    const uint kMask1 = 0x03030303u;
    const uint kMask2 = 0x0f0f0f0fu;
    const __global uchar* ps = blk + 96;
    uint aux0 = (uint)ps[0] | ((uint)ps[1] << 8) | ((uint)ps[2] << 16) | ((uint)ps[3] << 24);
    uint aux1 = (uint)ps[4] | ((uint)ps[5] << 8) | ((uint)ps[6] << 16) | ((uint)ps[7] << 24);
    uint aux2 = (uint)ps[8] | ((uint)ps[9] << 8) | ((uint)ps[10] << 16) | ((uint)ps[11] << 24);
    const uint tmp = aux2;
    aux2 = ((aux0 >> 4) & kMask2) | (((tmp >> 4) & kMask1) << 4);
    uint aux3 = ((aux1 >> 4) & kMask2) | (((tmp >> 6) & kMask1) << 4);
    aux0 = (aux0 & kMask2) | (((tmp >> 0) & kMask1) << 4);
    aux1 = (aux1 & kMask2) | (((tmp >> 2) & kMask1) << 4);

    __private float dl[16];
    __attribute__((opencl_unroll_hint))
    for (int i = 0; i < 4; ++i) {
        dl[i +  0] = d_all * (float)((int)(char)((aux0 >> (8 * i)) & 0xFFu) - 32);
        dl[i +  4] = d_all * (float)((int)(char)((aux1 >> (8 * i)) & 0xFFu) - 32);
        dl[i +  8] = d_all * (float)((int)(char)((aux2 >> (8 * i)) & 0xFFu) - 32);
        dl[i + 12] = d_all * (float)((int)(char)((aux3 >> (8 * i)) & 0xFFu) - 32);
    }

    // 3. 2 halves (n=0 / n=128) x 4 j-shifts, each writing 32 halfs.
    int o = 0;
    __attribute__((opencl_unroll_hint))
    for (int nh = 0; nh < 2; ++nh) {
        const __private uchar* qs_h = qs + 32 * nh;
        __attribute__((opencl_unroll_hint))
        for (int j = 0; j < 4; ++j) {
            const int   shift = 2 * j;
            const uchar mask  = (uchar)(1u << (4 * nh + j));
            const float dl_lo = dl[8 * nh + 2 * j + 0];
            const float dl_hi = dl[8 * nh + 2 * j + 1];
            __attribute__((opencl_unroll_hint))
            for (int l = 0; l < 16; ++l) {
                const int q = (int)((qs_h[l]      >> shift) & 3) - ((hmask[l]      & mask) ? 0 : 4);
                out[o + l]      = (half)(dl_lo * (float)q);
            }
            __attribute__((opencl_unroll_hint))
            for (int l = 0; l < 16; ++l) {
                const int q = (int)((qs_h[l + 16] >> shift) & 3) - ((hmask[l + 16] & mask) ? 0 : 4);
                out[o + l + 16] = (half)(dl_hi * (float)q);
            }
            o += 32;
        }
    }
}
#endif

#if defined(GGUF_IS_IQ3_XXS)
// Inner l-loop is fully unrolled so `7*l` / `2*l` / sign-bit masks become compile-time constants.
// The 8 stores per l target independent `out[]` slots, so no accumulator-chain break is needed
// here (unlike the streaming dot in fc_gguf_opt.cl which uses 4-way acc).
inline void FUNC(tq_decode_block)(const __global uchar* blk, __private half* out) {
    const float d = (float)FUNC_CALL(tq_load_f16)(blk);
    const __global uchar* qs = blk + 2;
    const __global uchar* scales_signs = blk + 2 + 64;
    for (int ib32 = 0; ib32 < 8; ++ib32) {
        const __global uchar* p4 = scales_signs + 4 * ib32;
        const uint  aux32 = (uint)p4[0] | ((uint)p4[1] << 8) | ((uint)p4[2] << 16) | ((uint)p4[3] << 24);
        const float db    = d * (0.5f + (float)(aux32 >> 28)) * 0.5f;
        const __global uchar* qsp = qs + 8 * ib32;
        const int ao_b = 32 * ib32;
        __attribute__((opencl_unroll_hint))
        for (int l = 0; l < 4; ++l) {
            const uchar signs = CONST_ARRAY_REF(ksigns_iq2xs)[(aux32 >> (7 * l)) & 127u];
            const uint  g1    = CONST_ARRAY_REF(iq3xxs_grid)[qsp[2 * l + 0]];
            const uint  g2    = CONST_ARRAY_REF(iq3xxs_grid)[qsp[2 * l + 1]];
            const int   ao    = ao_b + 8 * l;
            out[ao + 0] = (half)(db * (float)(uchar)( g1        & 0xFFu) * ((signs & 1u  ) ? -1.0f : 1.0f));
            out[ao + 1] = (half)(db * (float)(uchar)((g1 >>  8) & 0xFFu) * ((signs & 2u  ) ? -1.0f : 1.0f));
            out[ao + 2] = (half)(db * (float)(uchar)((g1 >> 16) & 0xFFu) * ((signs & 4u  ) ? -1.0f : 1.0f));
            out[ao + 3] = (half)(db * (float)(uchar)((g1 >> 24) & 0xFFu) * ((signs & 8u  ) ? -1.0f : 1.0f));
            out[ao + 4] = (half)(db * (float)(uchar)( g2        & 0xFFu) * ((signs & 16u ) ? -1.0f : 1.0f));
            out[ao + 5] = (half)(db * (float)(uchar)((g2 >>  8) & 0xFFu) * ((signs & 32u ) ? -1.0f : 1.0f));
            out[ao + 6] = (half)(db * (float)(uchar)((g2 >> 16) & 0xFFu) * ((signs & 64u ) ? -1.0f : 1.0f));
            out[ao + 7] = (half)(db * (float)(uchar)((g2 >> 24) & 0xFFu) * ((signs & 128u) ? -1.0f : 1.0f));
        }
    }
}
#endif

#if defined(GGUF_IS_IQ3_S)
// IQ3_S block layout (110 bytes / 256 elements; ggml block_iq3_s). See fc_gguf_opt.cl IQ3_S section
// for the field-level description. The decoded block is 256 halfs; downstream per-32 requant
// (REQUANT_GROUP=32) aligns one-to-one with the ib32 sub-block boundary (each ib32 sub-block of
// 32 elements has its own `db = d * (1 + 2*scale4)` sub-scale).

// Inner l-loop fully unrolled so `2*l` shifts collapse to compile-time constants for the
// 9-bit index assembly. Independent out[] writes -> no accumulator-chain break needed.
inline void FUNC(tq_decode_block)(const __global uchar* blk, __private half* out) {
    const float d = (float)FUNC_CALL(tq_load_f16)(blk);
    const __global uchar* qs     = blk + 2;
    const __global uchar* qh     = blk + 66;
    const __global uchar* signs  = blk + 74;
    const __global uchar* scales = blk + 106;
    for (int ib32 = 0; ib32 < 8; ++ib32) {
        const uchar sc_byte = scales[ib32 >> 1];
        const int   sc4     = (ib32 & 1) ? (int)(sc_byte >> 4) : (int)(sc_byte & 0xF);
        const float db      = d * (float)(1 + 2 * sc4);
        const uchar qhb     = qh[ib32];
        const __global uchar* qsp = qs + 8 * ib32;
        const __global uchar* sgp = signs + 4 * ib32;
        const int ao_b = 32 * ib32;
        __attribute__((opencl_unroll_hint))
        for (int l = 0; l < 4; ++l) {
            const uint  idx1 = (uint)qsp[2*l + 0] | ((((uint)qhb >> (2*l    )) & 1u) << 8);
            const uint  idx2 = (uint)qsp[2*l + 1] | ((((uint)qhb >> (2*l + 1)) & 1u) << 8);
            const uint  g1   = CONST_ARRAY_REF(iq3s_grid)[idx1];
            const uint  g2   = CONST_ARRAY_REF(iq3s_grid)[idx2];
            const uchar sb   = sgp[l];
            const int   ao   = ao_b + 8 * l;
            out[ao + 0] = (half)(db * (float)(uchar)( g1        & 0xFFu) * ((sb & 1u  ) ? -1.0f : 1.0f));
            out[ao + 1] = (half)(db * (float)(uchar)((g1 >>  8) & 0xFFu) * ((sb & 2u  ) ? -1.0f : 1.0f));
            out[ao + 2] = (half)(db * (float)(uchar)((g1 >> 16) & 0xFFu) * ((sb & 4u  ) ? -1.0f : 1.0f));
            out[ao + 3] = (half)(db * (float)(uchar)((g1 >> 24) & 0xFFu) * ((sb & 8u  ) ? -1.0f : 1.0f));
            out[ao + 4] = (half)(db * (float)(uchar)( g2        & 0xFFu) * ((sb & 16u ) ? -1.0f : 1.0f));
            out[ao + 5] = (half)(db * (float)(uchar)((g2 >>  8) & 0xFFu) * ((sb & 32u ) ? -1.0f : 1.0f));
            out[ao + 6] = (half)(db * (float)(uchar)((g2 >> 16) & 0xFFu) * ((sb & 64u ) ? -1.0f : 1.0f));
            out[ao + 7] = (half)(db * (float)(uchar)((g2 >> 24) & 0xFFu) * ((sb & 128u) ? -1.0f : 1.0f));
        }
    }
}
#endif

#if defined(GGUF_IS_IQ2_S)
// IQ2_S block layout (82 bytes / 256 elements; ggml block_iq2_s). See fc_gguf_opt.cl IQ2_S section
// for the field-level description. Note: each REQUANT_GROUP=32 = one ib32 sub-block carries TWO
// independent sub-scales db0/db1 (low/high nibble of scales[ib32]); the transcode_target() maps
// IQ2_S to i8 so the worst-case ~167:1 in-group dynamic range survives requantisation.
// l-loop peeled into l=0,1 (db0 for elems 0..15) and l=2,3 (db1 for elems 16..31) so the
// per-element `(l<2) ? db0 : db1` ternary becomes a compile-time constant per peeled body.
// Bit-exact vs the literal CPU port: identical FP ops in identical order, only the loop
// structure changes (each out[] slot receives the same value, written in the same sequence).
inline void FUNC(tq_decode_block)(const __global uchar* blk, __private half* out) {
    const float d = (float)FUNC_CALL(tq_load_f16)(blk);
    const __global uchar* qs     = blk + 2;
    const __global uchar* signs  = blk + 34;
    const __global uchar* qh     = blk + 66;
    const __global uchar* scales = blk + 74;
    for (int ib32 = 0; ib32 < 8; ++ib32) {
        const uchar sc  = scales[ib32];
        const float db0 = d * (0.5f + (float)(sc & 0xF)) * 0.25f;
        const float db1 = d * (0.5f + (float)(sc >> 4))  * 0.25f;
        const uchar qhb = qh[ib32];
        const __global uchar* qsp = qs    + 4 * ib32;
        const __global uchar* sgp = signs + 4 * ib32;
        const int ao_b = 32 * ib32;
        // l = 0, 1 -> elems 0..15 use db0.
        __attribute__((opencl_unroll_hint))
        for (int l = 0; l < 2; ++l) {
            const uint  idx = (uint)qsp[l] | (((uint)qhb << (8 - 2 * l)) & 0x300u);
            const ulong g   = CONST_ARRAY_REF(iq2s_grid)[idx];
            const uchar sb  = sgp[l];
            const int   ao  = ao_b + 8 * l;
            out[ao + 0] = (half)(db0 * (float)(uchar)( g        & 0xFFu) * ((sb & 1u  ) ? -1.0f : 1.0f));
            out[ao + 1] = (half)(db0 * (float)(uchar)((g >>  8) & 0xFFu) * ((sb & 2u  ) ? -1.0f : 1.0f));
            out[ao + 2] = (half)(db0 * (float)(uchar)((g >> 16) & 0xFFu) * ((sb & 4u  ) ? -1.0f : 1.0f));
            out[ao + 3] = (half)(db0 * (float)(uchar)((g >> 24) & 0xFFu) * ((sb & 8u  ) ? -1.0f : 1.0f));
            out[ao + 4] = (half)(db0 * (float)(uchar)((g >> 32) & 0xFFu) * ((sb & 16u ) ? -1.0f : 1.0f));
            out[ao + 5] = (half)(db0 * (float)(uchar)((g >> 40) & 0xFFu) * ((sb & 32u ) ? -1.0f : 1.0f));
            out[ao + 6] = (half)(db0 * (float)(uchar)((g >> 48) & 0xFFu) * ((sb & 64u ) ? -1.0f : 1.0f));
            out[ao + 7] = (half)(db0 * (float)(uchar)((g >> 56) & 0xFFu) * ((sb & 128u) ? -1.0f : 1.0f));
        }
        // l = 2, 3 -> elems 16..31 use db1.
        __attribute__((opencl_unroll_hint))
        for (int l = 2; l < 4; ++l) {
            const uint  idx = (uint)qsp[l] | (((uint)qhb << (8 - 2 * l)) & 0x300u);
            const ulong g   = CONST_ARRAY_REF(iq2s_grid)[idx];
            const uchar sb  = sgp[l];
            const int   ao  = ao_b + 8 * l;
            out[ao + 0] = (half)(db1 * (float)(uchar)( g        & 0xFFu) * ((sb & 1u  ) ? -1.0f : 1.0f));
            out[ao + 1] = (half)(db1 * (float)(uchar)((g >>  8) & 0xFFu) * ((sb & 2u  ) ? -1.0f : 1.0f));
            out[ao + 2] = (half)(db1 * (float)(uchar)((g >> 16) & 0xFFu) * ((sb & 4u  ) ? -1.0f : 1.0f));
            out[ao + 3] = (half)(db1 * (float)(uchar)((g >> 24) & 0xFFu) * ((sb & 8u  ) ? -1.0f : 1.0f));
            out[ao + 4] = (half)(db1 * (float)(uchar)((g >> 32) & 0xFFu) * ((sb & 16u ) ? -1.0f : 1.0f));
            out[ao + 5] = (half)(db1 * (float)(uchar)((g >> 40) & 0xFFu) * ((sb & 32u ) ? -1.0f : 1.0f));
            out[ao + 6] = (half)(db1 * (float)(uchar)((g >> 48) & 0xFFu) * ((sb & 64u ) ? -1.0f : 1.0f));
            out[ao + 7] = (half)(db1 * (float)(uchar)((g >> 56) & 0xFFu) * ((sb & 128u) ? -1.0f : 1.0f));
        }
    }
}
#endif

#if defined(GGUF_IS_IQ2_XS)
// IQ2_XS block layout (74 bytes / 256 elements; ggml block_iq2_xs). See fc_gguf_opt.cl IQ2_XS
// section for the field-level description. Note: each REQUANT_GROUP=32 = one ib32 sub-block
// carries TWO independent sub-scales db0/db1 (low/high nibble of scales[ib32]); the
// transcode_target() maps IQ2_XS to i8 (same tier as IQ2_S) so the worst-case ~167:1 in-group
// dynamic range survives requantisation.
// l-loop peeled (same idiom as IQ2_S transcode above): l=0,1 use db0, l=2,3 use db1.
// Bit-exact: identical FP ops in identical order, only loop structure changes.
inline void FUNC(tq_decode_block)(const __global uchar* blk, __private half* out) {
    const float d = (float)FUNC_CALL(tq_load_f16)(blk);
    const __global uchar* qs     = blk + 2;
    const __global uchar* scales = blk + 66;
    for (int ib32 = 0; ib32 < 8; ++ib32) {
        const uchar sc  = scales[ib32];
        const float db0 = d * (0.5f + (float)(sc & 0xF)) * 0.25f;
        const float db1 = d * (0.5f + (float)(sc >> 4))  * 0.25f;
        const __global uchar* qsp = qs + 8 * ib32;
        const int ao_b = 32 * ib32;
        // l = 0, 1 -> elems 0..15 use db0.
        __attribute__((opencl_unroll_hint))
        for (int l = 0; l < 2; ++l) {
            const uint  q   = (uint)qsp[2 * l] | ((uint)qsp[2 * l + 1] << 8);
            const ulong g   = CONST_ARRAY_REF(iq2xs_grid)[q & 0x1FFu];
            const uchar sb  = CONST_ARRAY_REF(ksigns_iq2xs)[(q >> 9) & 0x7Fu];
            const int   ao  = ao_b + 8 * l;
            out[ao + 0] = (half)(db0 * (float)(uchar)( g        & 0xFFu) * ((sb & 1u  ) ? -1.0f : 1.0f));
            out[ao + 1] = (half)(db0 * (float)(uchar)((g >>  8) & 0xFFu) * ((sb & 2u  ) ? -1.0f : 1.0f));
            out[ao + 2] = (half)(db0 * (float)(uchar)((g >> 16) & 0xFFu) * ((sb & 4u  ) ? -1.0f : 1.0f));
            out[ao + 3] = (half)(db0 * (float)(uchar)((g >> 24) & 0xFFu) * ((sb & 8u  ) ? -1.0f : 1.0f));
            out[ao + 4] = (half)(db0 * (float)(uchar)((g >> 32) & 0xFFu) * ((sb & 16u ) ? -1.0f : 1.0f));
            out[ao + 5] = (half)(db0 * (float)(uchar)((g >> 40) & 0xFFu) * ((sb & 32u ) ? -1.0f : 1.0f));
            out[ao + 6] = (half)(db0 * (float)(uchar)((g >> 48) & 0xFFu) * ((sb & 64u ) ? -1.0f : 1.0f));
            out[ao + 7] = (half)(db0 * (float)(uchar)((g >> 56) & 0xFFu) * ((sb & 128u) ? -1.0f : 1.0f));
        }
        // l = 2, 3 -> elems 16..31 use db1.
        __attribute__((opencl_unroll_hint))
        for (int l = 2; l < 4; ++l) {
            const uint  q   = (uint)qsp[2 * l] | ((uint)qsp[2 * l + 1] << 8);
            const ulong g   = CONST_ARRAY_REF(iq2xs_grid)[q & 0x1FFu];
            const uchar sb  = CONST_ARRAY_REF(ksigns_iq2xs)[(q >> 9) & 0x7Fu];
            const int   ao  = ao_b + 8 * l;
            out[ao + 0] = (half)(db1 * (float)(uchar)( g        & 0xFFu) * ((sb & 1u  ) ? -1.0f : 1.0f));
            out[ao + 1] = (half)(db1 * (float)(uchar)((g >>  8) & 0xFFu) * ((sb & 2u  ) ? -1.0f : 1.0f));
            out[ao + 2] = (half)(db1 * (float)(uchar)((g >> 16) & 0xFFu) * ((sb & 4u  ) ? -1.0f : 1.0f));
            out[ao + 3] = (half)(db1 * (float)(uchar)((g >> 24) & 0xFFu) * ((sb & 8u  ) ? -1.0f : 1.0f));
            out[ao + 4] = (half)(db1 * (float)(uchar)((g >> 32) & 0xFFu) * ((sb & 16u ) ? -1.0f : 1.0f));
            out[ao + 5] = (half)(db1 * (float)(uchar)((g >> 40) & 0xFFu) * ((sb & 32u ) ? -1.0f : 1.0f));
            out[ao + 6] = (half)(db1 * (float)(uchar)((g >> 48) & 0xFFu) * ((sb & 64u ) ? -1.0f : 1.0f));
            out[ao + 7] = (half)(db1 * (float)(uchar)((g >> 56) & 0xFFu) * ((sb & 128u) ? -1.0f : 1.0f));
        }
    }
}
#endif

// ---- main transcode kernel ----
// TRANSCODE_TO_I4 : 1 -> pack two i4 nibbles per output byte; 0 -> one i8 per byte.
// QMAX            : 7 (i4 symmetric) or 127 (i8 symmetric).
// REQUANT_GROUP   : elements sharing one f16 scale (divides GGUF_BLOCK_ELEM).
KERNEL(fc_gguf_transcode)(
    const __global uchar* W,        // GGUF block weights [N, K] (opaque bytes)
          __global uchar* WQ,       // out: packed low-bit weight [N, K] (i4 packed / i8)
          __global half*  SC        // out: per-group f16 scale [N, K/REQUANT_GROUP]
)
{
    const int n   = (int)get_global_id(0);          // output row (subgroup lane axis, padded to SG)
    const int blk = (int)get_global_id(1);          // GGUF block index along K
    const int blocks_per_row = K_SIZE / GGUF_BLOCK_ELEM;
    if (n >= N_SIZE || blk >= blocks_per_row)
        return;

    const __global uchar* w_row = W + (uint)n * (uint)blocks_per_row * GGUF_BLOCK_BYTES;

    // Decode the whole GGUF block ONCE. Every REQUANT group inside it reuses this decoded window, so
    // the expensive bit-unpacking runs a single time per block instead of once per group.
    half blk_vals[GGUF_BLOCK_ELEM];
    FUNC_CALL(tq_decode_block)(w_row + (uint)blk * GGUF_BLOCK_BYTES, blk_vals);

    const int groups_per_block = GGUF_BLOCK_ELEM / REQUANT_GROUP;
    const uint row_base = (uint)n * (uint)K_SIZE;
#if !TRANSCODE_TO_I4
    __global char* wq_i8 = (__global char*)WQ;
#endif

    // Symmetric per-group requantization for each REQUANT group within the decoded block.
    for (int gi = 0; gi < groups_per_block; ++gi) {
        const int off_in_blk = gi * REQUANT_GROUP;        // group offset within the decoded block
        const int g  = blk * groups_per_block + gi;       // global group index along K
        const int k0 = g * REQUANT_GROUP;                 // first K element of this group

        float amax = 0.0f;
        for (int i = 0; i < REQUANT_GROUP; ++i) {
            float v = fabs((float)blk_vals[off_in_blk + i]);
            amax = fmax(amax, v);
        }
        const float scale     = (amax > 0.0f) ? (amax / (float)QMAX) : 1.0f;
        const float inv_scale = (amax > 0.0f) ? ((float)QMAX / amax) : 0.0f;

        // Scale md is [K/group, N] (per-K-group x per-N): element (g, n) at g*N + n.
        SC[(uint)g * (uint)N_SIZE + (uint)n] = (half)scale;

#if TRANSCODE_TO_I4
        // i4 packed two-per-byte; weight byte index = (n*K + k)/2. REQUANT_GROUP is even.
        for (int i = 0; i < REQUANT_GROUP; i += 2) {
            const int k = k0 + i;
            int q0 = (int)round((float)blk_vals[off_in_blk + i]     * inv_scale);
            int q1 = (int)round((float)blk_vals[off_in_blk + i + 1] * inv_scale);
            q0 = clamp(q0, -8, 7);
            q1 = clamp(q1, -8, 7);
            const uint byte_idx = (row_base + (uint)k) >> 1; // two consecutive k share one byte
            WQ[byte_idx] = (uchar)((q0 & 0x0F) | ((q1 & 0x0F) << 4));
        }
#else
        for (int i = 0; i < REQUANT_GROUP; ++i) {
            int q = (int)round((float)blk_vals[off_in_blk + i] * inv_scale);
            q = clamp(q, -128, 127);
            wq_i8[row_base + (uint)(k0 + i)] = (char)q;
        }
#endif
    }
}
