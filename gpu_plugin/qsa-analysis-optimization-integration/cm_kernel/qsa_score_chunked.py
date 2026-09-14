"""Bounded QUERY-ROW Q2 scratch, never candidate-column chunks.

Planning is pure Python (no clops import/device needed). Construction allocates
one scratch and one concatenated, sequence-safe tile map, outside timing.
enqueue runs score(full history)->finalize for each chunk on the caller's single
in-order clops queue. Global iq, metadata, selected/counts remain caller-owned.
"""
from __future__ import annotations

from dataclasses import dataclass
from operator import index

DEFAULT_SCORE_BYTES = 128 * 1024 * 1024
_KERNELS = {}
FINALIZER_CHOICES = ("legacy", "scalar", "cooperative", "fast", "fast-wg", "fast-wg8", "fast-wg16",
                     "fast-wg16-cached")


def finalizer_spec(finalizer=None, *, cooperative=False, wg=16):
    """Explicit selection only; legacy/None retains the caller's old policy.

    fast-wg inherits wg; numbered aliases pin it regardless of prefill/decode.
    No shape/device performance extrapolation or automatic default change.
    """
    if wg not in (8, 16) or finalizer not in (None, *FINALIZER_CHOICES):
        raise ValueError("expected WG8/16 and a supported explicit finalizer")
    mode = ("cooperative" if cooperative else "scalar") if finalizer in (None, "legacy") else finalizer
    if mode == "fast-wg16-cached":
        mode, wg = "fast-wg", 16
    if mode in ("fast-wg8", "fast-wg16"):
        wg = int(mode.removeprefix("fast-wg"))
        mode = "fast-wg"
    name = {"scalar": "qsa_topk_finalization", "cooperative": "qsa_topk_cooperative",
            "fast": "qsa_topk_radix_fast", "fast-wg": "qsa_topk_radix_wg"}[mode]
    return name, wg if mode in ("cooperative", "fast-wg") else 1


def topk_cache_capacity(max_blocks):
    """WG16 keys/worker: 64-aligned full-row width, capped at 8 KiB/thread.

    Size from the padded score width, not query-row count or K. Actual causal
    rows whose aligned worker chunk exceeds this capacity scan globally instead.
    CM only scans SIMD64 regions: never emit a non-multiple of 64 (e.g. 65).
    """
    mb = index(max_blocks)
    if mb < 1:
        raise ValueError("positive max_blocks required")
    return min(2048, ((mb + 16 * 64 - 1) // (16 * 64)) * 64)


@dataclass(frozen=True)
class ScoreChunk:
    row_begin: int
    row_end: int
    tile_offset: int
    tile_count: int


@dataclass(frozen=True)
class ScorePlan:
    num_tokens: int
    max_blocks: int
    scratch_rows: int
    scratch_bytes: int
    allocation_bytes: int
    budget_bytes: int
    guard_elements: int
    chunks: tuple[ScoreChunk, ...]
    tile_map: tuple[tuple[int, int], ...]


def plan_score_chunks(subsequence_begins, max_blocks, *, budget_bytes=DEFAULT_SCORE_BYTES,
                      row_cap=None, guard_elements=0, unchunked=False):
    """Round DOWN to complete score rows; tiles never cross sequence/chunk ends.

    max_blocks is the FULL padded history width, a positive multiple of eight.
    Optional guards count against the budget. Empty batches dispatch nothing and
    need only a one-element dummy (plus optional guards), not a score row.
    Explicit unchunked comparison refuses allocations above the same budget.
    """
    begins = tuple(index(v) for v in subsequence_begins)
    mb, budget, guards = index(max_blocks), index(budget_bytes), index(guard_elements)
    cap = None if row_cap is None else index(row_cap)
    if (not begins or begins[0] != 0 or any(a > b for a, b in zip(begins, begins[1:]))
            or begins[-1] > 0x7fffffff or mb < 8 or mb % 8 or guards < 0
            or not 4 <= budget <= DEFAULT_SCORE_BYTES or (cap is not None and cap < 1)):
        raise ValueError("ordered i32 begins from zero, padded width>=8, budget=4..128MiB, row_cap>=1 required")
    nt = begins[-1]
    row_bytes = mb * 4
    # CM scorer byte offsets are uint32. Even custom budgets must not wrap.
    available = min(budget, 1 << 32) - guards * 4
    rows = min(nt, max(0, available // row_bytes))
    if nt and rows < 1:
        raise ValueError(f"score budget {budget} cannot fit one full row ({row_bytes} bytes) plus guards")
    if unchunked:
        if cap is not None:
            raise ValueError("unchunked comparison cannot use row_cap")
        if nt > rows:
            raise ValueError("unchunked scores exceed budget; use row chunks")
    elif cap is not None:
        rows = min(rows, cap)
    payload = rows * row_bytes
    allocation = max(4, payload + guards * 4)
    if allocation > budget:
        raise ValueError("score budget cannot fit empty-batch dummy/guards")
    chunks, tiles = [], []
    seq = 0
    for begin in range(0, nt, max(1, rows)):
        end, offset = min(nt, begin + rows), len(tiles)
        while seq + 1 < len(begins) and begins[seq + 1] <= begin:
            seq += 1
        s = seq
        while s + 1 < len(begins) and begins[s] < end:
            tiles.extend((s, t) for t in range(max(begin, begins[s]), min(end, begins[s + 1]), 16))
            s += 1
        chunks.append(ScoreChunk(begin, end, offset, len(tiles) - offset))
    return ScorePlan(nt, mb, rows, payload, allocation, budget, guards, tuple(chunks), tuple(tiles))


class ChunkedQ2:
    """Owns scorer/finalizer, reusable score scratch and immutable row plan.

    score()/finalize() expose a chunk for streaming validation only. enqueue()
    performs no allocations, copies, poison reset, or synchronization.
    """
    def __init__(self, cfg, batch, max_blocks, *, backend="dpas", cooperative=False,
                 wg=16, partition_blocks=16, dense_bypass=True,
                 budget_bytes=DEFAULT_SCORE_BYTES, row_cap=None, guard_elements=64,
                 unchunked=False, finalizer=None):
        import numpy as np
        import qsa_harness as H
        from clops import cl

        if backend not in ("dpas", "partition") or wg not in (8, 16) or partition_blocks < 1:
            raise ValueError("backend dpas/partition, WG8/16 and positive partition required")
        final_name, final_wg = finalizer_spec(finalizer, cooperative=cooperative, wg=wg)
        # Numbered aliases pin the finalizer WG, not the scorer's other options.
        if finalizer in ("fast-wg8", "fast-wg16", "fast-wg16-cached"):
            wg = final_wg
        if cfg.block_topk < 0:
            raise ValueError("K must be nonnegative")
        self.plan = plan_score_chunks(batch.subsequence_begins, max_blocks,
            budget_bytes=budget_bytes, row_cap=row_cap, guard_elements=guard_elements,
            unchunked=unchunked)
        self.backend, self.unchunked = backend, unchunked
        self.num_seqs, self.partition = batch.num_seqs, partition_blocks
        self.wg, self.dense_bypass = final_wg, dense_bypass
        self.score_name = "qsa_score_tile_dpas" if backend == "dpas" else "qsa_score_partition"
        self.final_name = final_name
        self.cache_capacity = topk_cache_capacity(self.plan.max_blocks) if finalizer == "fast-wg16-cached" else 0
        def build(name):
            # Keep all old modes' JITs unchanged; do not specialize the scorer
            # by cache size. The full JIT tuple separates cached/uncached builds.
            cache_jit = ({"QSA_TOPK_CACHE_CAPACITY": self.cache_capacity}
                         if name == self.final_name and self.cache_capacity else {})
            jit = cfg.jit(name, QSA_SCORE_ROW_CHUNK=int(not unchunked), QSA_SCORE_WG=16,
                QSA_TOPK_WG=wg, QSA_PARTITION_BLOCKS=partition_blocks,
                QSA_DENSE_BYPASS=int(dense_bypass), QSA_DENSE_SCORE_BYPASS=int(dense_bypass),
                **cache_jit)
            key = tuple(jit.items())
            if key not in _KERNELS:
                _KERNELS[key] = H.build_kernel(name + ".cm", jit)
            return _KERNELS[key]
        self.scorer, self.finalizer = build(self.score_name), build(self.final_name)
        tm = np.asarray(self.plan.tile_map, np.int32).reshape(-1, 2)
        self.tile_map = cl.tensor(tm if tm.size else np.zeros(1, np.int32))
        data = np.full(self.plan.allocation_bytes // 4, -777, np.float32)
        n = self.plan.scratch_bytes // 4
        data[:n] = self.poison().reshape(-1)
        self.scores = cl.tensor(data)

    def poison(self):
        import numpy as np
        a = np.full((self.plan.scratch_rows, self.plan.max_blocks), np.nan, np.float32)
        a[1::2] = -12345
        return a

    def read(self):
        import numpy as np
        a = self.scores.numpy().reshape(-1)
        n = self.plan.scratch_bytes // 4
        np.testing.assert_array_equal(a[n:], -777)
        return a[:n].reshape(self.plan.scratch_rows, self.plan.max_blocks).copy()

    def score(self, chunk, inputs, *, guard_groups=0):
        p, c = self.plan, chunk
        extra = () if self.unchunked else (c.row_begin, c.row_end)
        if self.backend == "dpas":
            self.scorer.enqueue(self.score_name, [(c.tile_count + guard_groups) * 16, 1, 1], [16, 1, 1],
                *inputs, self.tile_map, self.scores, c.tile_count, p.max_blocks,
                *extra, *((c.tile_offset,) if not self.unchunked else ()))
        else:
            parts = (p.max_blocks + self.partition - 1) // self.partition
            self.scorer.enqueue(self.score_name, [c.row_end - c.row_begin + guard_groups, 1, parts], [1, 1, 1],
                *inputs, self.scores, p.num_tokens, self.num_seqs, p.max_blocks, *extra)

    def finalize(self, chunk, past, begins, selected, counts, *, guard_groups=0):
        p, c = self.plan, chunk
        extra = () if self.unchunked else (c.row_begin, c.row_end)
        self.finalizer.enqueue(self.final_name, [(c.row_end - c.row_begin + guard_groups) * self.wg, 1, 1],
            [self.wg, 1, 1], self.scores, past, begins, selected, counts,
            p.num_tokens, self.num_seqs, p.max_blocks, *extra)

    def enqueue(self, inputs, selected, counts, *, guard_groups=0):
        for c in self.plan.chunks:
            self.score(c, inputs, guard_groups=guard_groups)
            self.finalize(c, inputs[2], inputs[3], selected, counts, guard_groups=guard_groups)