"""Explicit standalone scalar/split-selected-KV Q3 dispatch. Default stays scalar.

Use Q3Attention(cfg, partitions=P) to opt in, including two-dispatch P=1.
Caller owns FP32 scratch/output lifetime; enqueue performs no allocation/copy/finish.
Requires the same in-order queue for both dispatches (out-of-order needs dependencies).
Metadata contract is unchanged: valid complete block IDs, -1 padded slots,
0 <= selected_counts <= K, valid pages, and uint32 byte offsets without overflow.
No automatic routing or claim of speedup on unmeasured devices/shapes.
"""
from __future__ import annotations

import qsa_harness as H


class Q3Attention:
    def __init__(self, cfg, *, partitions=None):
        if partitions is not None and (type(partitions) is not int or not 1 <= partitions <= 16):
            raise ValueError("partitions must be None (scalar) or an integer in [1,16]")
        if (cfg.heads_num <= 0 or cfg.kv_heads_num <= 0 or
                cfg.heads_num % cfg.kv_heads_num or cfg.k_head_size <= 0 or
                cfg.v_head_size <= 0 or cfg.k_head_size % 16 or cfg.v_head_size % 16 or
                cfg.compress_ratio <= 0 or cfg.pa_block_size <= 0 or
                cfg.pa_block_size % cfg.compress_ratio or cfg.block_topk < 0):
            raise ValueError("unsupported Q3 head/page/selection geometry")
        self.cfg, self.partitions = cfg, partitions
        self.name = "qsa_sparse_attention" if partitions is None else "qsa_sparse_attention_split"
        extra = {} if partitions is None else {"QSA_SPLIT_PARTITIONS": partitions}
        self.kernel = H.build_kernel(self.name + ".cm", cfg.jit(self.name, **extra))
        self.reducer = None
        if partitions is not None:
            name = "qsa_sparse_attention_split_reduce"
            self.reducer = H.build_kernel(name + ".cm", cfg.jit(name, **extra))

    def scratch_elements(self, num_tokens):
        """Return (acc, ml) FP32 element counts; allocate dummy buffers for zeros."""
        if num_tokens < 0:
            raise ValueError("negative token count")
        rows = num_tokens * self.cfg.heads_num * (self.partitions or 0)
        sizes = rows * self.cfg.v_head_size, rows * 2
        if max(sizes) * 4 >= 2**32:
            raise ValueError("scratch exceeds uint32 byte-offset range")
        return sizes

    def enqueue(self, inputs, output, num_tokens, num_seqs, q_base, q_pitch,
                q_head_stride, gate_delta, *, partial_acc=None, partial_ml=None,
                guard_items=False):
        """inputs: query,K,V,past,begins,page_ids,page_begins,selected,counts."""
        if num_tokens < 0 or num_seqs < 0:
            raise ValueError("negative batch size")
        if num_tokens == 0:
            return
        if num_seqs == 0 or len(inputs) != 9:
            raise ValueError("nonempty Q3 requires sequence metadata and nine inputs")
        if min(q_base, q_pitch, q_head_stride, gate_delta) < 0 or any(
                x % 2 for x in (q_base, q_pitch, q_head_stride, gate_delta)):
            raise ValueError("Q layout offsets must be nonnegative and dword aligned")
        self.scratch_elements(num_tokens)
        hq = self.cfg.heads_num
        g = int(guard_items)
        layout = (num_tokens, num_seqs, q_base, q_pitch, q_head_stride, gate_delta)
        if self.partitions is None:
            self.kernel.enqueue(self.name, [hq + g, num_tokens + g, 1], [1, 1, 1],
                                *inputs, output, *layout)
        else:
            if partial_acc is None or partial_ml is None:
                raise ValueError("split Q3 requires caller-owned FP32 acc and ml scratch")
            self.kernel.enqueue(self.name, [hq + g, num_tokens + g, self.partitions + g],
                                [1, 1, 1], *inputs, partial_acc, partial_ml, *layout)
            self.reducer.enqueue("qsa_sparse_attention_split_reduce",
                [hq + g, num_tokens + g, 1], [1, 1, 1], inputs[0], partial_acc, partial_ml,
                output, num_tokens, q_base, q_pitch, q_head_stride, gate_delta)