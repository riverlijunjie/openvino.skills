"""Shared plumbing for the standalone QSA CM kernel tests.

* ``build_kernel`` flattens a ``.cm`` file (inlining the local ``"..."`` includes
  exactly the way the plugin codegen does) and compiles it through clops with the
  right JIT ``#define``s and the production build options.
* ``bench`` times a kernel launch with GPU profiling, warmup and a min-of-N.
* ``PagedBatch`` builds the paged-attention metadata (past_lens,
  subsequence_begins, block tables) that almost every QSA kernel consumes.

Run every test from the ``cm_kernel/`` directory so ``from clops import cl`` picks
up the local package. The kernels only compile/run on the remote BMG GPU.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
KERNEL_DIR = os.path.join(HERE, "kernels")
INCLUDE_DIR = os.path.join(HERE, "include")

# Same options the plugin uses (see get_qsa_build_options in qsa.cpp).
QSA_BUILD_OPTIONS = "-cmc -Qxcm_register_file_size=256"

_INCLUDE_RE = re.compile(r'^\s*#\s*include\s*"([^"]+)"\s*$')


def _inline_includes(path: str, seen: set[str]) -> str:
    """Recursively inline local ``#include "..."`` directives, once per file.

    System includes (``#include <...>``) are left untouched - they are provided by
    the CM frontend. Including each local header exactly once mimics ``#pragma
    once``; every QSA header is idempotent so this is equivalent to the codegen's
    flattened single-compilation-unit output.
    """
    real = os.path.realpath(path)
    if real in seen:
        return ""
    seen.add(real)
    out = []
    with open(real) as fh:
        for line in fh:
            m = _INCLUDE_RE.match(line)
            if m:
                inc = os.path.join(INCLUDE_DIR, m.group(1))
                if os.path.isfile(inc):
                    out.append(f"// >>> inlined {m.group(1)}\n")
                    out.append(_inline_includes(inc, seen))
                    out.append(f"// <<< end {m.group(1)}\n")
                    continue
            out.append(line)
    return "".join(out)


def flatten_source(kernel_file: str, defines: dict) -> str:
    """Return the flat CM source: JIT defines + the kernel with includes inlined."""
    path = kernel_file if os.path.isabs(kernel_file) else os.path.join(KERNEL_DIR, kernel_file)
    header = "".join(f"#define {k} {v}\n" for k, v in defines.items())
    body = _inline_includes(path, set())
    return header + "\n" + body


def build_kernel(kernel_file: str, defines: dict, *, dump_dir: str = "", extra_opts: str = ""):
    """Compile a QSA kernel and return the clops kernels handle."""
    from clops import cl

    src = flatten_source(kernel_file, defines)
    opts = (QSA_BUILD_OPTIONS + " " + extra_opts).strip()
    return cl.kernels(src, opts, dump_dir)


# ---------------------------------------------------------------------------
# timing
# ---------------------------------------------------------------------------
# timing, cache flush and roofline
# ---------------------------------------------------------------------------

# Intel Arc B580 (BMG) peak numbers, see remote_machine.txt:
#   20 Xe cores * 2048 f16 FLOP/clk * 2.90 GHz = 118.784 TFLOP/s
#   memory bandwidth 456 GB/s.
PEAK_FLOPS = 118.784e12
PEAK_BW = 456e9
L2_BYTES = 18 * 1024 * 1024   # BMG last-level cache; flush buffer must exceed it


class CacheFlusher:
    """Evicts the GPU last-level cache between timed iterations.

    Reads+writes a buffer several times the size of the cache so that a kernel's
    inputs are cold on the next launch, removing cache-hit interference from the
    measurement. The flush runs in its own finish() so its time is never counted.
    """

    def __init__(self, mb: int = 96):
        from clops import cl
        self.n = (mb * 1024 * 1024) // 4
        self.buf = cl.tensor(np.zeros(self.n, np.int32))
        src = r"""
        __kernel void cache_flush(__global int* buf) {
            int i = get_global_id(0);
            buf[i] = buf[i] + 1;
        }
        """
        self.kern = cl.kernels(src, "")

    def flush(self):
        from clops import cl
        self.kern.enqueue("cache_flush", [self.n], [256], self.buf)
        cl.finish()


def _lat_sum(lat):
    return float(sum(lat)) if isinstance(lat, (list, tuple)) else float(lat)


def bench(enqueue_fn, *, iters: int = 100, warmup: int = 10, flusher: "CacheFlusher | None" = None):
    """Time ``enqueue_fn`` and return stats in ms.

    Each measured iteration is preceded by a cache flush (in its own finish, so it
    is not timed), then the kernel runs and is finished on its own. Returns a dict
    with mean / min / median / std over ``iters`` samples (ns from clops -> ms).
    """
    from clops import cl

    cl.profiling(True)
    for _ in range(warmup):
        enqueue_fn()
    cl.finish()

    samples = []
    for _ in range(iters):
        if flusher is not None:
            flusher.flush()
        enqueue_fn()
        samples.append(_lat_sum(cl.finish()))
    a = np.array(samples, dtype=np.float64) * 1e-6  # ns -> ms
    return {"mean": float(a.mean()), "min": float(a.min()),
            "median": float(np.median(a)), "std": float(a.std())}


def roofline_ms(flops: float, byts: float):
    """Return (roofline_ms, bound) for the BMG peak numbers."""
    t_c = flops / PEAK_FLOPS
    t_m = byts / PEAK_BW
    return max(t_c, t_m) * 1e3, ("compute" if t_c >= t_m else "memory")


def report(name: str, phase: str, size: int, stats: dict, flops: float, byts: float):
    """Print one measured-vs-roofline row, tagged with the phase (prefill/decode)."""
    rl, bound = roofline_ms(flops, byts)
    meas = stats["mean"]
    eff = 100.0 * rl / meas if meas > 0 else 0.0
    dim = "q" if phase == "prefill" else "past"
    print(f"[{name:>16} {phase:<7}] {dim}={size:<5} mean={meas:8.4f} min={stats['min']:8.4f} "
          f"median={stats['median']:8.4f} ms | GFLOP={flops/1e9:8.3f} MB={byts/1e6:8.2f} "
          f"| roofline={rl:8.4f} ms ({bound:7}) | eff={eff:5.1f}%")


def sweep(sizes, make_case, *, phase: str = "prefill", iters: int = 100, warmup: int = 10,
          flush: bool = True, do_check: bool = False):
    """Run ``make_case(n)`` for each n in sizes and print a roofline row for each.

    ``phase`` is "prefill" (n = query tokens, no history) or "decode" (n = KV history
    length, one new token). ``make_case(n)`` returns {name, enqueue, flops, bytes,
    check(optional)}; ``check`` (slow pure-python ref) only runs when ``do_check``.
    """
    flusher = CacheFlusher() if flush else None
    for n in sizes:
        case = make_case(n)
        if do_check and case.get("check"):
            case["check"]()
        stats = bench(case["enqueue"], iters=iters, warmup=warmup, flusher=flusher)
        report(case["name"], phase, n, stats, case["flops"], case["bytes"])


# ---------------------------------------------------------------------------
# tensors
# ---------------------------------------------------------------------------

def f16(arr) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(arr, dtype=np.float16))


def i32(arr) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(arr, dtype=np.int32))


def rand_f16(rng: np.random.Generator, *shape, scale: float = 1.0, bias: float = 0.0) -> np.ndarray:
    """Match the gtest data distribution: values in [-1,1] at 1/16 resolution."""
    q = rng.integers(-16, 17, size=shape).astype(np.float32) / 16.0
    return f16(q * scale + bias)


# ---------------------------------------------------------------------------
# paged-attention batch metadata
# ---------------------------------------------------------------------------

@dataclass
class PagedBatch:
    """Paged-attention page tables + per-sequence lengths for a QSA batch.

    seqs is a list of (past_len, q_len). Pages are allocated sequentially across
    the batch into a single physical pool; ``block_indices`` maps each sequence's
    logical page to its physical page (optionally shuffled to exercise the
    indirection), exactly as PagedAttention feeds the QSA kernels.
    """
    pa_block: int
    past_lens: np.ndarray            # [num_seqs] i32
    subsequence_begins: np.ndarray   # [num_seqs+1] i32
    block_indices: np.ndarray        # [used_pages] i32
    block_indices_begins: np.ndarray # [num_seqs+1] i32
    num_blocks: int                  # physical pages allocated
    seqs: list[tuple[int, int]]

    @property
    def num_seqs(self) -> int:
        return len(self.seqs)

    @property
    def num_tokens(self) -> int:
        return int(self.subsequence_begins[-1])

    def seq_of_token(self, tok: int) -> tuple[int, int]:
        """Return (seq, abs_pos) for a global token index (causal position)."""
        for s, (past, _q) in enumerate(self.seqs):
            b, e = int(self.subsequence_begins[s]), int(self.subsequence_begins[s + 1])
            if b <= tok < e:
                return s, past + (tok - b)
        raise IndexError(tok)


def make_selection(batch: "PagedBatch", r: int, block_topk: int, *, seed: int = 0):
    """Synthesize plausible top-k selection metadata for a standalone Q3 test.

    For each token, the visible COMPLETE blocks are [0, abs_pos // r). We pick up
    to ``block_topk`` distinct ids in ascending order (the radix selector emits
    ascending ids) and pad the rest with -1. Returns
    (selected_blocks[num_tokens, block_topk] i32, selected_counts[num_tokens] i32).
    """
    rng = np.random.default_rng(seed)
    nt = batch.num_tokens
    sel = np.full((nt, block_topk), -1, dtype=np.int32)
    cnt = np.zeros(nt, dtype=np.int32)
    for tok in range(nt):
        _s, abs_pos = batch.seq_of_token(tok)
        n_complete = (abs_pos + 1) // r
        k = min(block_topk, n_complete)
        if k > 0:
            chosen = np.sort(rng.choice(n_complete, size=k, replace=False))
            sel[tok, :k] = chosen.astype(np.int32)
        cnt[tok] = k
    return sel, cnt


def make_score_tile_map(batch: "PagedBatch", tile: int = 16):
    """(seq, token_begin) pairs for the prefill dpas tiles, mirroring qsa.cpp."""
    rows = []
    for s, (_past, q) in enumerate(batch.seqs):
        begin = int(batch.subsequence_begins[s])
        for t in range(0, q, tile):
            rows.append((s, begin + t))
    return i32(rows).reshape(-1, 2), len(rows)


def ceil8(n: int) -> int:
    return ((int(n) + 7) // 8) * 8


def make_prepare_map(batch: "PagedBatch", r: int):
    """(seq, qsa_block_id) pairs for every QSA block touched by new tokens (Q1)."""
    rows = []
    for s, (past, q) in enumerate(batch.seqs):
        total = past + q
        first = past // r
        last = (total - 1) // r
        for b in range(first, last + 1):
            rows.append((s, b))
    return i32(rows).reshape(-1, 2), len(rows)


def make_indexer_inputs(cfg, batch: "PagedBatch", *, seed: int = 0):
    """Random indexer queries + block summaries for the Q2 scoring kernels.

    Returns (indexer_q[num_tokens, Hidx, Di] f16, summary_cache[num_blocks*spp*Di] f16,
    max_blocks, n_complete[num_tokens]).
    """
    rng = np.random.default_rng(seed)
    Hidx, Di, spp = cfg.indexer_heads, cfg.indexer_head_dim, cfg.summaries_per_page
    iq = rand_f16(rng, batch.num_tokens, Hidx, Di, scale=0.25)
    summ = rand_f16(rng, batch.num_blocks * spp * Di, scale=0.25)
    ncomp = np.array([(batch.seq_of_token(t)[1] + 1) // cfg.compress_ratio
                      for t in range(batch.num_tokens)], dtype=np.int64)
    max_blocks = max(8, ceil8(int(ncomp.max()) if len(ncomp) else 0))
    return iq, summ, max_blocks, ncomp


def make_mode_batch(mode: str, size: int, pa_block: int, decode_batch: int = 1) -> "PagedBatch":
    """Build a prefill or decode batch swept by ``size``.

    prefill: one sequence of ``size`` query tokens, no history (past = 0).
    decode:  ``decode_batch`` sequences, each one new token on top of ``size`` tokens
             of KV history (the decode "history token" count the request refers to).
    """
    if mode == "prefill":
        return make_batch([(0, size)], pa_block)
    return make_batch([(size, 1)] * decode_batch, pa_block)


def make_batch(seqs: list[tuple[int, int]], pa_block: int, *,
               shuffle_pages: bool = True, seed: int = 0) -> PagedBatch:
    past_lens, sub_begins, blk_begins, blk_indices = [], [0], [0], []
    phys = 0
    for past, q in seqs:
        total = past + q
        pages = (total + pa_block - 1) // pa_block
        for _ in range(pages):
            blk_indices.append(phys)
            phys += 1
        past_lens.append(past)
        sub_begins.append(sub_begins[-1] + q)
        blk_begins.append(len(blk_indices))
    blk_indices = np.array(blk_indices, dtype=np.int32)
    if shuffle_pages and len(blk_indices) > 1:
        np.random.default_rng(seed).shuffle(blk_indices)
    return PagedBatch(
        pa_block=pa_block,
        past_lens=i32(past_lens),
        subsequence_begins=i32(sub_begins),
        block_indices=blk_indices,
        block_indices_begins=i32(blk_begins),
        num_blocks=phys,
        seqs=list(seqs),
    )
