# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
GGUF block model + NumPy reference decoders for the extracted OV GPU GGUF kernels.

Single source of truth for:
  * block geometry  (block_elem_count / block_byte_size) -- mirrors OV core
    element_type.hpp gguf_block_elem_count / gguf_block_byte_size exactly.
  * raw-block byte generators (produce *valid* random GGUF blocks for any format;
    every random byte pattern decodes in-range, see note below).
  * NumPy dequantizers that mirror the .cl decoders bit-for-bit (same field offsets,
    same unpack math), used as the correctness oracle for the GPU kernels.

Why "any random bytes are valid": for every supported format the grid/codebook index
is masked to exactly the table size (e.g. IQ2_XS uses q & 0x1FF -> [0,511] with a
512-entry grid; IQ3_S uses qs|qh-bit -> [0,511] with a 512-entry grid), so a fully
random qs/sign/mask region never indexes out of range. Only the f16 `d`/`dmin` scale
fields are overwritten with small controlled half values so the dequantized weights
(and the f16 matmul) stay in a sane numeric range.

The IQ codebook/sign tables are parsed directly from the verbatim kernel header
kernels/gguf/gguf_iq_tables.hpp so there is exactly one copy of the tables.
"""

import os
import re
import numpy as np

# --------------------------------------------------------------------------------------
# Block geometry -- must match OV core element_type.hpp (gguf_block_*_count / *_byte_size)
# --------------------------------------------------------------------------------------
# name -> (block_elem_count, block_byte_size)
GGUF_BLOCK_GEOM = {
    "gguf_q4_0":   (32, 18),
    "gguf_q8_0":   (32, 34),
    "gguf_q3_k":   (256, 110),
    "gguf_q4_k":   (256, 144),
    "gguf_q5_k":   (256, 176),
    "gguf_q6_k":   (256, 210),
    "gguf_iq2_xs": (256, 74),
    "gguf_iq2_s":  (256, 82),
    "gguf_iq3_xxs":(256, 98),
    "gguf_iq3_s":  (256, 110),
}

# JIT flag (GGUF_IS_*) selecting the per-format decoder -- mirrors gguf_type_jit_flag().
GGUF_TYPE_FLAG = {
    "gguf_q4_0":   "GGUF_IS_Q4_0",
    "gguf_q8_0":   "GGUF_IS_Q8_0",
    "gguf_q4_k":   "GGUF_IS_Q4_K",
    "gguf_q5_k":   "GGUF_IS_Q5_K",
    "gguf_q6_k":   "GGUF_IS_Q6_K",
    "gguf_q3_k":   "GGUF_IS_Q3_K",
    "gguf_iq2_xs": "GGUF_IS_IQ2_XS",
    "gguf_iq2_s":  "GGUF_IS_IQ2_S",
    "gguf_iq3_xxs":"GGUF_IS_IQ3_XXS",
    "gguf_iq3_s":  "GGUF_IS_IQ3_S",
}

# transcode target -- mirrors transcode_target(): (to_i4, qmax)
GGUF_TRANSCODE_TARGET = {
    "gguf_q4_0":   (True, 7),
    "gguf_q4_k":   (True, 7),
    "gguf_q3_k":   (True, 7),
    "gguf_iq3_xxs":(True, 7),
    "gguf_iq3_s":  (True, 7),
    "gguf_q5_k":   (False, 127),
    "gguf_q6_k":   (False, 127),
    "gguf_q8_0":   (False, 127),
    "gguf_iq2_xs": (False, 127),
    "gguf_iq2_s":  (False, 127),
}

GGUF_REQUANT_GROUP = 32  # mirrors GGUF_REQUANT_GROUP in fc_gguf_opt.cpp


def block_elem(name):
    return GGUF_BLOCK_GEOM[name][0]


def block_bytes(name):
    return GGUF_BLOCK_GEOM[name][1]


# --------------------------------------------------------------------------------------
# IQ table parsing (from the verbatim kernel header) -- one source of truth.
# --------------------------------------------------------------------------------------
_TABLE_CACHE = {}

def _iq_tables_path():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "..", "kernels", "gguf", "gguf_iq_tables.hpp")


def load_iq_table(name):
    """Parse CONST_ARRAY_DECL(<name>) = { 0x.., 0x.., ... }; from the kernel header."""
    if name in _TABLE_CACHE:
        return _TABLE_CACHE[name]
    with open(_iq_tables_path(), "r") as f:
        text = f.read()
    # The header is wrapped in #if defined(GGUF_IS_*) guards; a few tables (ksigns_iq2xs)
    # appear under more than one guard. Parse every definition and keep the longest, so the
    # active-guard duplication does not matter. Values may be hex (0x..) or decimal.
    best = None
    for m in re.finditer(r"CONST_ARRAY_DECL\(\s*%s\s*\)\s*=\s*\{(.*?)\}\s*;" % re.escape(name),
                         text, re.DOTALL):
        body = m.group(1)
        # match either 0x-prefixed hex or bare decimal integers (handles both table styles)
        toks = re.findall(r"0x[0-9a-fA-F]+|\b\d+\b", body)
        vals = [int(v, 16) if v.lower().startswith("0x") else int(v) for v in toks]
        if best is None or len(vals) > len(best):
            best = vals
    if not best:
        raise KeyError("table %r not found in gguf_iq_tables.hpp" % name)
    arr = np.array(best, dtype=np.uint64)
    _TABLE_CACHE[name] = arr
    return arr


# --------------------------------------------------------------------------------------
# f16 helpers
# --------------------------------------------------------------------------------------
def f16_bytes(value):
    """Return the 2 little-endian bytes of an f16 holding `value`."""
    h = np.float16(value)
    return np.frombuffer(h.tobytes(), dtype=np.uint8).copy()


def load_f16_le(b, off):
    """Decode 2 little-endian bytes at b[off:off+2] as a python float (mirrors gguf_load_f16)."""
    bits = np.uint16(b[off]) | (np.uint16(b[off + 1]) << np.uint16(8))
    return float(np.frombuffer(np.uint16(bits).tobytes(), dtype=np.float16)[0])


# --------------------------------------------------------------------------------------
# Raw-block generation. Returns weight bytes of shape [N, blocks_per_row * block_bytes]
# laid out exactly as the kernel reads them: w_row = W + n*blocks_per_row*block_bytes.
# `scale_amp` controls the magnitude of the f16 d/dmin fields so dequant stays sane.
# --------------------------------------------------------------------------------------
def _rng(seed):
    return np.random.default_rng(seed)


def _set_d_fields(blk, offsets, rng, amp):
    """Overwrite the listed f16 scale-field byte offsets with small random halves."""
    for off in offsets:
        v = np.float16(rng.uniform(-amp, amp))
        # avoid exact zero so amax>0 paths are exercised
        if float(v) == 0.0:
            v = np.float16(amp)
        blk[off:off + 2] = np.frombuffer(v.tobytes(), dtype=np.uint8)


# f16 scale-field offsets per format (everything else is left as random bytes)
_D_FIELD_OFFSETS = {
    "gguf_q4_0":   [0],          # d
    "gguf_q8_0":   [0],          # d
    "gguf_q4_k":   [0, 2],       # d, dmin
    "gguf_q5_k":   [0, 2],       # d, dmin
    "gguf_q6_k":   [208],        # d (after 128+64+16)
    "gguf_q3_k":   [108],        # d_all
    "gguf_iq2_xs": [0],          # d
    "gguf_iq2_s":  [0],          # d
    "gguf_iq3_xxs":[0],          # d
    "gguf_iq3_s":  [0],          # d
}


def gen_weight_bytes(name, N, K, seed=1234, scale_amp=0.05):
    """Generate a valid random GGUF weight matrix [N,K] as packed bytes (np.uint8)."""
    be, bb = GGUF_BLOCK_GEOM[name]
    assert K % be == 0, "K=%d not a multiple of block_elem=%d" % (K, be)
    bpr = K // be
    rng = _rng(seed)
    W = rng.integers(0, 256, size=(N, bpr, bb), dtype=np.uint8)
    offs = _D_FIELD_OFFSETS[name]
    # For IQ3_XXS the per-ib32 scale lives in the top nibble of aux32 (bytes 2+64..), and the
    # f16 d is at [0]; random bytes there are fine but keep d small. Same handled by offs.
    for n in range(N):
        for kb in range(bpr):
            _set_d_fields(W[n, kb], offs, rng, scale_amp)
    return W.reshape(N, bpr * bb)


# --------------------------------------------------------------------------------------
# NumPy reference dequantizers. Each returns float32 [N, K] dequantized weights, mirroring
# the corresponding FUNC(gguf_block_dot) / tq_decode_block in the .cl exactly.
# --------------------------------------------------------------------------------------
def _blocks(W, name):
    be, bb = GGUF_BLOCK_GEOM[name]
    N = W.shape[0]
    bpr = W.shape[1] // bb
    return N, bpr, be, bb, W.reshape(N, bpr, bb)


def deq_q4_0(W):
    N, bpr, be, bb, B = _blocks(W, "gguf_q4_0")
    out = np.zeros((N, bpr * be), np.float32)
    for n in range(N):
        for kb in range(bpr):
            blk = B[n, kb]
            d = load_f16_le(blk, 0)
            qs = blk[2:18].astype(np.int32)
            base = kb * be
            for j in range(16):
                lo = (qs[j] & 0x0F) - 8
                hi = (qs[j] >> 4) - 8
                out[n, base + j] = lo * d
                out[n, base + j + 16] = hi * d
    return out


def deq_q8_0(W):
    N, bpr, be, bb, B = _blocks(W, "gguf_q8_0")
    out = np.zeros((N, bpr * be), np.float32)
    for n in range(N):
        for kb in range(bpr):
            blk = B[n, kb]
            d = load_f16_le(blk, 0)
            qs = blk[2:34].view(np.int8).astype(np.float32)
            out[n, kb * be:kb * be + 32] = qs * d
    return out


def _scale_min_k4(j, q):
    if j < 4:
        d = q[j] & 63
        m = q[j + 4] & 63
    else:
        d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4)
        m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4)
    return int(d), int(m)


def deq_q4_k(W):
    N, bpr, be, bb, B = _blocks(W, "gguf_q4_k")
    out = np.zeros((N, bpr * be), np.float32)
    for n in range(N):
        for kb in range(bpr):
            blk = B[n, kb]
            d = load_f16_le(blk, 0)
            dmin = load_f16_le(blk, 2)
            scales = blk[4:16]
            qs = blk[16:144].astype(np.int32)
            base = kb * be
            o = 0
            is_ = 0
            qoff = 0
            for j in range(0, 256, 64):
                sc, m = _scale_min_k4(is_ + 0, scales); d1 = d * sc; m1 = dmin * m
                sc, m = _scale_min_k4(is_ + 1, scales); d2 = d * sc; m2 = dmin * m
                for l in range(32):
                    out[n, base + o] = d1 * float(qs[qoff + l] & 0x0F) - m1; o += 1
                for l in range(32):
                    out[n, base + o] = d2 * float(qs[qoff + l] >> 4) - m2; o += 1
                qoff += 32; is_ += 2
    return out


def deq_q5_k(W):
    N, bpr, be, bb, B = _blocks(W, "gguf_q5_k")
    out = np.zeros((N, bpr * be), np.float32)
    for n in range(N):
        for kb in range(bpr):
            blk = B[n, kb]
            d = load_f16_le(blk, 0)
            dmin = load_f16_le(blk, 2)
            scales = blk[4:16]
            qh = blk[16:48].astype(np.int32)
            ql = blk[48:176].astype(np.int32)
            base = kb * be
            o = 0; is_ = 0; qloff = 0; u1 = 1; u2 = 2
            for j in range(0, 256, 64):
                sc, m = _scale_min_k4(is_ + 0, scales); d1 = d * sc; m1 = dmin * m
                sc, m = _scale_min_k4(is_ + 1, scales); d2 = d * sc; m2 = dmin * m
                for l in range(32):
                    q = (ql[qloff + l] & 0x0F) + (16 if (qh[l] & u1) else 0)
                    out[n, base + o] = d1 * float(q) - m1; o += 1
                for l in range(32):
                    q = (ql[qloff + l] >> 4) + (16 if (qh[l] & u2) else 0)
                    out[n, base + o] = d2 * float(q) - m2; o += 1
                qloff += 32; is_ += 2; u1 <<= 2; u2 <<= 2
    return out


def deq_q6_k(W):
    N, bpr, be, bb, B = _blocks(W, "gguf_q6_k")
    out = np.zeros((N, bpr * be), np.float32)
    for n in range(N):
        for kb in range(bpr):
            blk = B[n, kb]
            ql = blk[0:128].astype(np.int32)
            qh = blk[128:192].astype(np.int32)
            sc = blk[192:208].view(np.int8).astype(np.int32)
            d = load_f16_le(blk, 208)
            base = kb * be
            o = 0; qloff = 0; qhoff = 0; scoff = 0
            for nn in range(0, 256, 128):
                for l in range(32):
                    is_ = l // 16
                    q1 = ((ql[qloff + l + 0] & 0x0F) | (((qh[qhoff + l] >> 0) & 3) << 4)) - 32
                    q2 = ((ql[qloff + l + 32] & 0x0F) | (((qh[qhoff + l] >> 2) & 3) << 4)) - 32
                    q3 = ((ql[qloff + l + 0] >> 4) | (((qh[qhoff + l] >> 4) & 3) << 4)) - 32
                    q4 = ((ql[qloff + l + 32] >> 4) | (((qh[qhoff + l] >> 6) & 3) << 4)) - 32
                    out[n, base + o + l + 0] = d * float(sc[scoff + is_ + 0]) * q1
                    out[n, base + o + l + 32] = d * float(sc[scoff + is_ + 2]) * q2
                    out[n, base + o + l + 64] = d * float(sc[scoff + is_ + 4]) * q3
                    out[n, base + o + l + 96] = d * float(sc[scoff + is_ + 6]) * q4
                o += 128; qloff += 64; qhoff += 32; scoff += 8
    return out


def deq_q3_k(W):
    N, bpr, be, bb, B = _blocks(W, "gguf_q3_k")
    out = np.zeros((N, bpr * be), np.float32)
    for n in range(N):
        for kb in range(bpr):
            blk = B[n, kb]
            hmask = blk[0:32].astype(np.int32)
            qs = blk[32:96].astype(np.int32)
            d_all = load_f16_le(blk, 108)
            ps = blk[96:108].astype(np.uint32)
            kMask1 = np.uint32(0x03030303); kMask2 = np.uint32(0x0f0f0f0f)
            aux0 = np.uint32(ps[0] | (ps[1] << 8) | (ps[2] << 16) | (ps[3] << 24))
            aux1 = np.uint32(ps[4] | (ps[5] << 8) | (ps[6] << 16) | (ps[7] << 24))
            aux2 = np.uint32(ps[8] | (ps[9] << 8) | (ps[10] << 16) | (ps[11] << 24))
            tmp = aux2
            aux2 = ((aux0 >> np.uint32(4)) & kMask2) | (((tmp >> np.uint32(4)) & kMask1) << np.uint32(4))
            aux3 = ((aux1 >> np.uint32(4)) & kMask2) | (((tmp >> np.uint32(6)) & kMask1) << np.uint32(4))
            aux0 = (aux0 & kMask2) | (((tmp >> np.uint32(0)) & kMask1) << np.uint32(4))
            aux1 = (aux1 & kMask2) | (((tmp >> np.uint32(2)) & kMask1) << np.uint32(4))
            dl = np.zeros(16, np.float32)
            for i in range(4):
                dl[i + 0] = d_all * float(np.int8((aux0 >> np.uint32(8 * i)) & np.uint32(0xFF)) - 32)
                dl[i + 4] = d_all * float(np.int8((aux1 >> np.uint32(8 * i)) & np.uint32(0xFF)) - 32)
                dl[i + 8] = d_all * float(np.int8((aux2 >> np.uint32(8 * i)) & np.uint32(0xFF)) - 32)
                dl[i + 12] = d_all * float(np.int8((aux3 >> np.uint32(8 * i)) & np.uint32(0xFF)) - 32)
            base = kb * be
            o = 0
            for nh in range(2):
                qs_h = qs[32 * nh: 32 * nh + 32]
                hm_h = hmask  # hmask indexed l and l+16
                for j in range(4):
                    shift = 2 * j
                    mask = 1 << (4 * nh + j)
                    dl_lo = dl[8 * nh + 2 * j + 0]
                    dl_hi = dl[8 * nh + 2 * j + 1]
                    for l in range(16):
                        q = ((qs_h[l] >> shift) & 3) - (0 if (hmask[l] & mask) else 4)
                        out[n, base + o + l] = dl_lo * float(q)
                    for l in range(16):
                        q = ((qs_h[l + 16] >> shift) & 3) - (0 if (hmask[l + 16] & mask) else 4)
                        out[n, base + o + l + 16] = dl_hi * float(q)
                    o += 32
    return out


def deq_iq3_xxs(W):
    grid = load_iq_table("iq3xxs_grid").astype(np.uint32)
    ksigns = load_iq_table("ksigns_iq2xs").astype(np.uint32)
    N, bpr, be, bb, B = _blocks(W, "gguf_iq3_xxs")
    out = np.zeros((N, bpr * be), np.float32)
    for n in range(N):
        for kb in range(bpr):
            blk = B[n, kb]
            d = load_f16_le(blk, 0)
            qs = blk[2:66].astype(np.uint32)
            ss = blk[66:98].astype(np.uint32)
            base = kb * be
            for ib32 in range(8):
                p4 = ss[4 * ib32:4 * ib32 + 4]
                aux32 = np.uint32(p4[0] | (p4[1] << 8) | (p4[2] << 16) | (p4[3] << 24))
                db = d * (0.5 + float(aux32 >> np.uint32(28))) * 0.5
                qsp = qs[8 * ib32:8 * ib32 + 8]
                ai_b = 32 * ib32
                for l in range(4):
                    signs = ksigns[int((aux32 >> np.uint32(7 * l)) & np.uint32(127))]
                    g1 = grid[int(qsp[2 * l + 0])]
                    g2 = grid[int(qsp[2 * l + 1])]
                    ai = ai_b + 8 * l
                    for t in range(4):
                        mag = float((g1 >> np.uint32(8 * t)) & np.uint32(0xFF))
                        sgn = -1.0 if (signs & (1 << t)) else 1.0
                        out[n, base + ai + t] = db * mag * sgn
                    for t in range(4):
                        mag = float((g2 >> np.uint32(8 * t)) & np.uint32(0xFF))
                        sgn = -1.0 if (signs & (1 << (4 + t))) else 1.0
                        out[n, base + ai + 4 + t] = db * mag * sgn
    return out


def deq_iq3_s(W):
    grid = load_iq_table("iq3s_grid").astype(np.uint32)
    N, bpr, be, bb, B = _blocks(W, "gguf_iq3_s")
    out = np.zeros((N, bpr * be), np.float32)
    for n in range(N):
        for kb in range(bpr):
            blk = B[n, kb]
            d = load_f16_le(blk, 0)
            qs = blk[2:66].astype(np.uint32)
            qh = blk[66:74].astype(np.uint32)
            signs = blk[74:106].astype(np.uint32)
            scales = blk[106:110].astype(np.uint32)
            base = kb * be
            for ib32 in range(8):
                sc_byte = scales[ib32 >> 1]
                sc4 = (sc_byte >> 4) if (ib32 & 1) else (sc_byte & 0xF)
                db = d * float(1 + 2 * int(sc4))
                qhb = qh[ib32]
                qsp = qs[8 * ib32:8 * ib32 + 8]
                sgp = signs[4 * ib32:4 * ib32 + 4]
                ai_b = 32 * ib32
                for l in range(4):
                    idx1 = int(qsp[2 * l + 0]) | (((int(qhb) >> (2 * l)) & 1) << 8)
                    idx2 = int(qsp[2 * l + 1]) | (((int(qhb) >> (2 * l + 1)) & 1) << 8)
                    g1 = grid[idx1]; g2 = grid[idx2]
                    sb = int(sgp[l]); ai = ai_b + 8 * l
                    for t in range(4):
                        mag = float((g1 >> np.uint32(8 * t)) & np.uint32(0xFF))
                        out[n, base + ai + t] = db * mag * (-1.0 if (sb & (1 << t)) else 1.0)
                    for t in range(4):
                        mag = float((g2 >> np.uint32(8 * t)) & np.uint32(0xFF))
                        out[n, base + ai + 4 + t] = db * mag * (-1.0 if (sb & (1 << (4 + t))) else 1.0)
    return out


def deq_iq2_s(W):
    grid = load_iq_table("iq2s_grid").astype(np.uint64)
    N, bpr, be, bb, B = _blocks(W, "gguf_iq2_s")
    out = np.zeros((N, bpr * be), np.float32)
    for n in range(N):
        for kb in range(bpr):
            blk = B[n, kb]
            d = load_f16_le(blk, 0)
            qs = blk[2:34].astype(np.uint32)
            signs = blk[34:66].astype(np.uint32)
            qh = blk[66:74].astype(np.uint32)
            scales = blk[74:82].astype(np.uint32)
            base = kb * be
            for ib32 in range(8):
                sc = scales[ib32]
                db0 = d * (0.5 + float(sc & 0xF)) * 0.25
                db1 = d * (0.5 + float(sc >> 4)) * 0.25
                qhb = int(qh[ib32])
                qsp = qs[4 * ib32:4 * ib32 + 4]
                sgp = signs[4 * ib32:4 * ib32 + 4]
                ai_b = 32 * ib32
                for l in range(4):
                    db = db0 if l < 2 else db1
                    idx = int(qsp[l]) | ((qhb << (8 - 2 * l)) & 0x300)
                    g = int(grid[idx]); sb = int(sgp[l]); ai = ai_b + 8 * l
                    for t in range(8):
                        mag = float((g >> (8 * t)) & 0xFF)
                        out[n, base + ai + t] = db * mag * (-1.0 if (sb & (1 << t)) else 1.0)
    return out


def deq_iq2_xs(W):
    grid = load_iq_table("iq2xs_grid").astype(np.uint64)
    ksigns = load_iq_table("ksigns_iq2xs").astype(np.uint32)
    N, bpr, be, bb, B = _blocks(W, "gguf_iq2_xs")
    out = np.zeros((N, bpr * be), np.float32)
    for n in range(N):
        for kb in range(bpr):
            blk = B[n, kb]
            d = load_f16_le(blk, 0)
            qs = blk[2:66].astype(np.uint32)
            scales = blk[66:74].astype(np.uint32)
            base = kb * be
            for ib32 in range(8):
                sc = scales[ib32]
                db0 = d * (0.5 + float(sc & 0xF)) * 0.25
                db1 = d * (0.5 + float(sc >> 4)) * 0.25
                qsp = qs[8 * ib32:8 * ib32 + 8]
                ai_b = 32 * ib32
                for l in range(4):
                    db = db0 if l < 2 else db1
                    q = int(qsp[2 * l]) | (int(qsp[2 * l + 1]) << 8)
                    g = int(grid[q & 0x1FF]); sb = int(ksigns[(q >> 9) & 0x7F]); ai = ai_b + 8 * l
                    for t in range(8):
                        mag = float((g >> (8 * t)) & 0xFF)
                        out[n, base + ai + t] = db * mag * (-1.0 if (sb & (1 << t)) else 1.0)
    return out


DEQUANT = {
    "gguf_q4_0": deq_q4_0,
    "gguf_q8_0": deq_q8_0,
    "gguf_q4_k": deq_q4_k,
    "gguf_q5_k": deq_q5_k,
    "gguf_q6_k": deq_q6_k,
    "gguf_q3_k": deq_q3_k,
    "gguf_iq3_xxs": deq_iq3_xxs,
    "gguf_iq3_s": deq_iq3_s,
    "gguf_iq2_s": deq_iq2_s,
    "gguf_iq2_xs": deq_iq2_xs,
}


def dequantize(name, W):
    """Dequantize packed weight bytes [N, bpr*bb] -> float32 [N, K]."""
    return DEQUANT[name](W)


if __name__ == "__main__":
    # smoke test: every format decodes to the right shape with finite values
    for name in GGUF_BLOCK_GEOM:
        be = block_elem(name)
        K = be * 2
        N = 3
        W = gen_weight_bytes(name, N, K, seed=7)
        D = dequantize(name, W)
        assert D.shape == (N, K), (name, D.shape)
        assert np.all(np.isfinite(D)), name
        print("%-14s block_elem=%3d block_bytes=%3d  deq range [% .4f, % .4f]"
              % (name, be, block_bytes(name), D.min(), D.max()))
