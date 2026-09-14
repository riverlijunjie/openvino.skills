"""Explicit opt-in two-dispatch Q1; the existing scalar driver stays the default.

Caller owns a contiguous float32 [num_tokens, (Hidx+1)*Di] scratch tensor and
enqueues on the same in-order queue. No hidden allocation/copy/finish is timed.
No qbase/strided tensor support: token indices use the existing prepare ABI.
"""
from __future__ import annotations

import qsa_harness as H


class PrepareDPAS:
    def __init__(self, cfg, wg=16):
        self.j = (cfg.indexer_heads + 1) * cfg.indexer_head_dim
        if (cfg.hidden_size < 32 or cfg.hidden_size % 32 or self.j % 8
                or cfg.indexer_head_dim % 16 or cfg.indexer_heads + 1 > wg
                or cfg.compress_ratio < 1 or cfg.pa_block_size % cfg.compress_ratio):
            raise ValueError("Unsupported Q1 DPAS shape; use the default scalar prepare")
        self.project = H.build_kernel("qsa_project_dpas.cm", cfg.jit("qsa_project_dpas"))
        self.prepare = H.build_kernel("qsa_prepare.cm", cfg.jit(
            "qsa_prepare_projected", QSA_PREPARE_WG=wg, QSA_PREPROJECTED=1))

    def enqueue(self, projected, num_tokens, gws, lws, *prepare_args, guard_tiles=False):
        """prepare_args is exactly the unchanged scalar kernel argument list."""
        if num_tokens == 0:
            return
        extra = int(guard_tiles)
        self.project.enqueue("qsa_project_dpas",
                             [self.j // 8 + extra, (num_tokens + 15) // 16 + extra, 1],
                             [1, 1, 1], prepare_args[0], prepare_args[1], projected, num_tokens)
        self.prepare.enqueue("qsa_prepare_projected", gws, lws, projected, *prepare_args)