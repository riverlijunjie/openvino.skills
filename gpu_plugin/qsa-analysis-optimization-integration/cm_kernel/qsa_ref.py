"""Numpy references for the QSA attention kernels (fp32 math, f16 inputs)."""

from __future__ import annotations

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sparse_attention_ref(cfg, batch, query, key_cache, value_cache,
                         selected_blocks, selected_counts, *, q_head_stride, gate_delta):
    """Reference for qsa_sparse_attention(.dpas): scaled-dot-product softmax over the
    selected complete QSA blocks plus the always-visible incomplete tail.

    query is [num_tokens, q_pitch] with per-head layout [q(Dh) | gate(Dv)] when the
    output gate is on. Returns output [num_tokens, Hq*Dv] float32.
    """
    Hq, Dh, Dv, r, pab = cfg.heads_num, cfg.k_head_size, cfg.v_head_size, cfg.compress_ratio, cfg.pa_block_size
    n_rep = cfg.n_rep
    K_TOPK = selected_blocks.shape[1]
    q = query.astype(np.float32).reshape(-1)
    kc = key_cache.astype(np.float32).reshape(batch.num_blocks, cfg.kv_heads_num, pab, Dh)
    vc = value_cache.astype(np.float32).reshape(batch.num_blocks, cfg.kv_heads_num, pab, Dv)
    out = np.zeros((batch.num_tokens, Hq * Dv), dtype=np.float32)

    for tok in range(batch.num_tokens):
        seq, abs_pos = batch.seq_of_token(tok)
        page_begin = int(batch.block_indices_begins[seq])
        n_complete = (abs_pos + 1) // r

        def kv_at(pos, kv_head):
            page = int(batch.block_indices[page_begin + pos // pab])
            tip = pos % pab
            return kc[page, kv_head, tip], vc[page, kv_head, tip]

        for head in range(Hq):
            kv_head = head // n_rep
            base = tok * (Hq * q_head_stride) + head * q_head_stride
            qh = q[base:base + Dh] * cfg.scale

            positions = []
            cnt = int(selected_counts[tok])
            for i in range(min(cnt, K_TOPK)):
                b = int(selected_blocks[tok, i])
                if b < 0:
                    continue
                positions.extend(range(b * r, b * r + r))
            positions.extend(range(n_complete * r, abs_pos + 1))

            m, s = -1e30, 0.0
            acc = np.zeros(Dv, dtype=np.float32)
            for pos in positions:
                k, v = kv_at(pos, kv_head)
                score = float(np.dot(qh, k))
                nm = max(m, score)
                corr = np.exp(m - nm)
                w = np.exp(score - nm)
                s = s * corr + w
                acc = acc * corr + v * w
                m = nm
            o = acc / s if s > 0 else acc
            if gate_delta:
                g = q[base + gate_delta: base + gate_delta + Dv]
                o = o * sigmoid(g)
            out[tok, head * Dv:(head + 1) * Dv] = o
    return out


def n_complete_of(abs_pos, r):
    return (abs_pos + 1) // r


def indexer_scores_ref(cfg, batch, indexer_q, summary_cache, max_blocks):
    """Reference block scores for the Q2 kernels.

    score(b) = (1/sqrt(Di)) * sum_h relu(q_h . summary_b), for b in [0, n_complete).
    Returns (scores[num_tokens, max_blocks] float32, n_complete[num_tokens] int).
    """
    Hidx, Di, r = cfg.indexer_heads, cfg.indexer_head_dim, cfg.compress_ratio
    spp = cfg.summaries_per_page
    q = indexer_q.astype(np.float32).reshape(batch.num_tokens, Hidx, Di)
    sc = summary_cache.astype(np.float32).reshape(-1)
    scores = np.full((batch.num_tokens, max_blocks), -np.inf, dtype=np.float32)
    ncomp = np.zeros(batch.num_tokens, dtype=np.int64)
    for tok in range(batch.num_tokens):
        seq, abs_pos = batch.seq_of_token(tok)
        page_begin = int(batch.block_indices_begins[seq])
        nc = n_complete_of(abs_pos, r)
        ncomp[tok] = nc
        for b in range(nc):
            page = int(batch.block_indices[page_begin + b // spp])
            off = (page * spp + (b % spp)) * Di
            summ = sc[off:off + Di]
            total = 0.0
            for h in range(Hidx):
                dot = float(np.dot(q[tok, h], summ))
                total += dot if dot > 0.0 else 0.0
            scores[tok, b] = total * cfg.indexer_scale
    return scores, ncomp


def topk_select_ref(scores_row, n_complete, K):
    """Ascending ids of the K highest scores among [0, n_complete); count=min(K,nc)."""
    nc = int(n_complete)
    if nc <= 0:
        return np.array([], dtype=np.int64), 0
    if nc <= K:
        return np.arange(nc, dtype=np.int64), nc
    idx = np.argpartition(scores_row[:nc], nc - K)[-K:]
    return np.sort(idx), K


def selection_matches(got_blocks, got_count, scores_row, n_complete, K, *, tol=1e-4):
    """True if the selection is a valid top-k (tolerating ties at the boundary score)."""
    ref_ids, cnt = topk_select_ref(scores_row, n_complete, K)
    got = set(int(x) for x in got_blocks[:got_count] if x >= 0)
    ref = set(int(x) for x in ref_ids)
    if got == ref:
        return True
    if got_count != cnt:
        return False
    # allow ties: every differing id must score within tol of the kth boundary score
    boundary = float(np.sort(scores_row[:int(n_complete)])[-K])
    for b in got ^ ref:
        if abs(float(scores_row[b]) - boundary) > tol:
            return False
    return True


def _rms_norm(v, w, eps):
    inv = 1.0 / np.sqrt(np.mean(v * v) + eps)
    return v * inv * w


def _rope(v, table, pos, rot):
    v = v.copy()
    half = rot // 2
    base = pos * rot
    for j in range(half):
        c = float(table[base + 2 * j]); s = float(table[base + 2 * j + 1])
        x0, x1 = v[j], v[j + half]
        v[j] = x0 * c - x1 * s
        v[j + half] = x1 * c + x0 * s
    return v


def prepare_ref(cfg, batch, mixed, weight, q_norm, k_norm, rotary, position_ids):
    """Reference for Q1 qsa_prepare.

    Returns (indexer_q[num_tokens, Hidx, Di], summaries) where summaries is a dict
    {(page, slot): vec[Di]} keyed the same way the summary_cache is addressed.
    """
    Hidx, Di, D = cfg.indexer_heads, cfg.indexer_head_dim, cfg.hidden_size
    ROT, r, spp = cfg.indexer_rotary_dim, cfg.compress_ratio, cfg.summaries_per_page
    eps = cfg.indexer_norm_eps
    W = weight.astype(np.float32).reshape((Hidx + 1) * Di, D)
    x = mixed.astype(np.float32).reshape(batch.num_tokens, D)
    proj = x @ W.T                      # [num_tokens, (Hidx+1)*Di]
    qn = q_norm.astype(np.float32); kn = k_norm.astype(np.float32)
    tbl = rotary.astype(np.float32).reshape(-1)

    iq = np.zeros((batch.num_tokens, Hidx, Di), dtype=np.float32)
    for t in range(batch.num_tokens):
        for h in range(Hidx):
            q = proj[t, h * Di:(h + 1) * Di].copy()
            q = _rms_norm(q, qn, eps)
            q = _rope(q, tbl, int(position_ids[t]), ROT)
            iq[t, h] = q

    raw_k = proj[:, Hidx * Di:(Hidx + 1) * Di]     # per-token raw key, no norm/rope
    summaries = {}
    for s, (past, q) in enumerate(batch.seqs):
        total = past + q
        page_begin = int(batch.block_indices_begins[s])
        seq_begin = int(batch.subsequence_begins[s])
        n_complete = total // r                      # blocks with all r slots filled
        for b in range(n_complete):
            # a block only produced here if it contains a NEW token of this sequence
            if (b + 1) * r <= past:
                continue
            pooled = np.zeros(Di, dtype=np.float32)
            for slot in range(r):
                p = b * r + slot
                tok = seq_begin + (p - past)
                pooled += raw_k[tok]
            pooled /= r
            pooled = _rms_norm(pooled, kn, eps)
            pooled = _rope(pooled, tbl, b * r, ROT)
            page = int(batch.block_indices[page_begin + b // spp])
            summaries[(page, b % spp)] = pooled
    return iq, summaries

