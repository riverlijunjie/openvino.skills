"""Opt-in standalone Q2 selection/partition helpers; existing defaults unchanged.

Caller owns buffers and in-order queue ordering. No allocation/copy/finish in
enqueue. Dense score bypass is allowed only with dense_bypass=True in BOTH
producer and finalizer. No automatic device/shape dispatch policy is installed.
"""
from __future__ import annotations

import qsa_harness as H


class TopKFinalizer:
    def __init__(self, cfg, *, cooperative=False, wg=16, dense_bypass=True):
        if cfg.block_topk < 0 or wg not in (8, 16):
            raise ValueError("top-k >= 0 and WG 8/16 required")
        self.wg = wg if cooperative else 1
        self.name = "qsa_topk_cooperative" if cooperative else "qsa_topk_finalization"
        self.kernel = H.build_kernel(self.name + ".cm", cfg.jit(
            self.name, QSA_TOPK_WG=wg, QSA_DENSE_BYPASS=int(dense_bypass)))

    def enqueue(self, scores, past, begins, selected, counts, num_tokens,
                num_seqs, max_blocks, *, guard_groups=0):
        if num_tokens == 0:
            return
        self.kernel.enqueue(self.name, [(num_tokens + guard_groups) * self.wg, 1, 1],
                            [self.wg, 1, 1], scores, past, begins, selected, counts,
                            num_tokens, num_seqs, max_blocks)


class PartitionScorer:
    """Only partition granularity changes; NOT cooperative score computation."""
    def __init__(self, cfg, *, partition_blocks=512, dense_score_bypass=False):
        if partition_blocks < 1:
            raise ValueError("partition_blocks must be positive")
        self.partition_blocks = partition_blocks
        self.name = "qsa_score_partition"
        self.kernel = H.build_kernel(self.name + ".cm", cfg.jit(
            self.name, QSA_PARTITION_BLOCKS=partition_blocks,
            QSA_DENSE_BYPASS=1, QSA_DENSE_SCORE_BYPASS=int(dense_score_bypass)))

    def enqueue(self, inputs, scores, num_tokens, num_seqs, max_blocks):
        if num_tokens == 0:
            return
        parts = (max_blocks + self.partition_blocks - 1) // self.partition_blocks
        self.kernel.enqueue(self.name, [num_tokens, 1, max(1, parts)], [1, 1, 1],
                            *inputs, scores, num_tokens, num_seqs, max_blocks)