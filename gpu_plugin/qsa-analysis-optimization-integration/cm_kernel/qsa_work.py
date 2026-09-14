"""Per-kernel FLOP and byte counts for the QSA roofline comparison.

FLOPs count multiply-adds x2. Bytes count the essential DRAM traffic the algorithm
must move (inputs read + outputs written, with GQA / summary reuse assumed ideal),
so the roofline is a hardware lower bound to compare the measured time against.
All f16 tensors are 2 bytes, i32/f32 are 4.
"""

from __future__ import annotations

import numpy as np

F16, I32, F32 = 2, 4, 4


def _abs_positions(batch):
    return np.array([batch.seq_of_token(t)[1] for t in range(batch.num_tokens)], dtype=np.int64)


def n_complete(batch, r):
    return (_abs_positions(batch) + 1) // r


def attended_positions(batch, cnt, r):
    """Positions each token attends in Q3 = selected blocks (r each) + causal tail."""
    ap = _abs_positions(batch)
    nc = (ap + 1) // r
    tail = (ap + 1) - nc * r          # (abs_pos+1) % r
    return cnt.astype(np.int64) * r + tail


def num_summaries(batch, r):
    """Complete QSA blocks that produce a summary across the batch."""
    tot = 0
    for past, q in batch.seqs:
        tot += (past + q) // r - (past // r if past else 0)
    return tot


def blocks_scored_unique(batch, r):
    """Distinct complete blocks any token scores (= summaries read once)."""
    return sum((past + q) // r for past, q in batch.seqs)


# --- per-kernel (flops, bytes) ------------------------------------------------

def q0(cfg, batch):
    nt, Hkv, Dh, Dv = batch.num_tokens, cfg.kv_heads_num, cfg.k_head_size, cfg.v_head_size
    byts = 2 * nt * Hkv * (Dh + Dv) * F16     # read K,V + write cache K,V
    return 0.0, float(byts)


def q1(cfg, batch):
    nt, D, Hidx, Di = batch.num_tokens, cfg.hidden_size, cfg.indexer_heads, cfg.indexer_head_dim
    J = (Hidx + 1) * Di
    flops = nt * J * D * 2                     # Q/K projection dominates
    nsumm = num_summaries(batch, cfg.compress_ratio)
    byts = (nt * D * F16                       # mixed
            + J * D * F16                      # weight (streamed once, ideal)
            + nt * Hidx * Di * F16             # indexer_q
            + nt * Di * F16                    # raw_k
            + nsumm * Di * F16)                # summaries
    return float(flops), float(byts)


def q2_score(cfg, batch, ncomp):
    """Scoring kernels (fused / dpas / partition): sum_h relu(q.summary).

    Roofline bytes = minimal unique DRAM traffic: each indexer query once, each
    block summary once (not re-read per token), scores written once.
    """
    nt, Hidx, Di = batch.num_tokens, cfg.indexer_heads, cfg.indexer_head_dim
    blocks = int(ncomp.sum())
    nsumm = blocks_scored_unique(batch, cfg.compress_ratio)
    flops = blocks * Hidx * Di * 2
    byts = (nt * Hidx * Di * F16          # indexer_q, read once
            + nsumm * Di * F16            # each summary read once (ideal L2 reuse)
            + blocks * F32)               # block_scores written
    return float(flops), float(byts)


def q2_topk(cfg, batch, ncomp):
    nt, K = batch.num_tokens, cfg.block_topk
    blocks = int(ncomp.sum())
    byts = blocks * F32 + nt * K * I32 + nt * I32   # read scores, write selected + counts
    return 0.0, float(byts)


def q3(cfg, batch, cnt, *, q_head_stride, kv_reuse_heads):
    """Sparse attention. Roofline bytes = minimal unique DRAM traffic: each KV
    position read once (ideal cache reuse across the tokens that share it), query
    and output once. kv_reuse_heads = Hkv (GQA-shared).
    """
    nt, Hq, Dh, Dv = batch.num_tokens, cfg.heads_num, cfg.k_head_size, cfg.v_head_size
    P = attended_positions(batch, cnt, cfg.compress_ratio)
    totP = int(P.sum())
    ap = _abs_positions(batch)
    union_positions = int(ap.max()) + 1 if len(ap) else 0   # distinct KV positions touched
    flops = Hq * totP * (Dh + Dv) * 2                        # QK (Dh) + PV (Dv)
    byts = (kv_reuse_heads * union_positions * (Dh + Dv) * F16   # K + V, each once
            + nt * Hq * q_head_stride * F16                      # query (+gate)
            + nt * Hq * Dv * F16)                               # output
    return float(flops), float(byts)
