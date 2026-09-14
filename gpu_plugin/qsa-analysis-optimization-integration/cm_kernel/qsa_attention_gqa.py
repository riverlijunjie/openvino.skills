"""Stage5 explicit opt-in only: one-token/all-GQA-heads N16 DPAS, W4/step16.

Q3GQAAttention(cfg) retains scalar. Pass enabled=True to select the new kernel.
head_major=False and vector_meta=False retain the original GQA path. Opt in with
enabled=True, head_major=True, vector_meta=True for the measured PTL winner
(about 1.35x at 64K / 1.23x at 128K in the standalone synthetic Q3 benchmark).
head_major changes launch order; vector_meta batches masked metadata loads.
Neither changes the score reduction/softmax arithmetic or enables auto routing.
Unsupported geometry raises; fallback='scalar' explicitly permits scalar instead.
No enqueue-time allocations, copies, finish, auto performance routing, or scratch.
Token map: caller-owned int32 [num_tokens,2] (sequence, global token), one entry
per token, e.g. H.make_score_tile_map(batch, tile=1). Map must not duplicate tokens.
Selected lists must be sorted, unique, complete/causal IDs and counts in [0,K].
All tensors/page tables and uint32 address ranges remain caller contracts.
"""
from __future__ import annotations

import qsa_harness as H
from qsa_attention_split import Q3Attention


class Q3GQAAttention:
    @staticmethod
    def unsupported_reason(cfg):
        if (cfg.heads_num <= 0 or cfg.kv_heads_num <= 0 or
                cfg.heads_num % cfg.kv_heads_num or not 1 <= cfg.n_rep <= 16):
            return "requires integral GQA group in [1,16]"
        if (cfg.k_head_size <= 0 or cfg.v_head_size <= 0 or
                cfg.k_head_size % 128 or cfg.v_head_size % 128):
            return "requires W4 Dh/Dv slices aligned to 32 half elements"
        # Limit accumulator pressure to the verified standalone range.
        if max(cfg.k_head_size, cfg.v_head_size) > 256:
            return "head dimensions above 256 are not validated"
        if (cfg.compress_ratio not in (2, 4, 8, 16) or cfg.pa_block_size <= 0 or
                cfg.pa_block_size % cfg.compress_ratio or cfg.block_topk < 0):
            return "requires r2/4/8/16, page divisible by r, nonnegative topk"
        if not cfg.is_causal:
            return "noncausal QSA is unsupported"
        return None

    def __init__(self, cfg, *, enabled=False, fallback="error",
                 vector_meta=False, head_major=False):
        if type(head_major) is not bool:
            raise ValueError("head_major must be bool")
        self.head_major = head_major
        if type(vector_meta) is not bool:
            raise ValueError("vector_meta must be bool")
        self.vector_meta = vector_meta
        if type(enabled) is not bool or fallback not in ("error", "scalar"):
            raise ValueError("enabled must be bool; fallback must be error or scalar")
        if not cfg.is_causal:
            raise ValueError("noncausal QSA is unsupported, including fallback")
        self.cfg = cfg
        self.reason = self.unsupported_reason(cfg) if enabled else "not opted in"
        if enabled and self.reason and fallback == "error":
            raise ValueError(self.reason)
        self.enabled = enabled and self.reason is None
        self.scalar = None
        self.name = "qsa_sparse_attention_gqa_dpas"
        if self.enabled:
            jit = cfg.jit(self.name, QSA_GQA_VECTOR_META=int(vector_meta),
                          QSA_GQA_HEAD_MAJOR=int(head_major))
            self.kernel = H.build_kernel(self.name + ".cm", jit)
        else:
            self.scalar = Q3Attention(cfg)

    def enqueue(self, inputs, output, num_tokens, num_seqs, q_base, q_pitch,
                q_head_stride, gate_delta, *, token_map=None, guard_items=False):
        layout = (num_tokens, num_seqs, q_base, q_pitch, q_head_stride, gate_delta)
        if any(type(x) is not int or not 0 <= x < 2**32 for x in layout):
            raise ValueError("batch sizes and Q layout must be uint32 integers")
        if type(guard_items) is not bool:
            raise ValueError("guard_items must be bool")
        if num_tokens == 0:
            return
        if inputs is None or len(inputs) != 9 or num_seqs == 0 or output is None:
            raise ValueError("nine inputs, sequences and output required")
        if self.scalar is not None:
            self.scalar.enqueue(inputs, output, num_tokens, num_seqs, q_base, q_pitch,
                                q_head_stride, gate_delta, guard_items=guard_items)
            return
        if token_map is None:
            raise ValueError("one-entry-per-token map required")
        # Conservative 2D surface base/pitch alignment; gate loads need dwords.
        if any(
                x % 16 for x in (q_base, q_pitch, q_head_stride)) or gate_delta % 2:
            raise ValueError("Q base/pitches must be 32-byte aligned; gate delta dword aligned")
        width = max(self.cfg.k_head_size,
                    gate_delta + self.cfg.v_head_size if self.cfg.has_output_gate else 0)
        if q_head_stride < width or q_pitch < self.cfg.heads_num * q_head_stride:
            raise ValueError("Q/gate storage overlaps heads or tokens")
        if max(q_base + num_tokens * q_pitch,
               num_tokens * self.cfg.heads_num * self.cfg.v_head_size) * 2 >= 2**32:
            raise ValueError("Q/output exceeds uint32 byte-offset range")
        g = int(guard_items)
        grid = ([(num_tokens + g) * 4, self.cfg.kv_heads_num + g, 1] if self.head_major else
                [(self.cfg.kv_heads_num + g) * 4, num_tokens + g, 1])
        self.kernel.enqueue(self.name, grid,
            [4, 1, 1], *inputs, token_map, output, num_tokens,
            q_base, q_pitch, q_head_stride, gate_delta)