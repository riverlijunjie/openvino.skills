"""QSA parameters derived from the model config.json.

The plugin builds every QSA CM kernel with a set of JIT ``#define``s (see
``make_qsa_jit`` in ``qsa.cpp``). This module reproduces those constants from the
HuggingFace ``config.json`` so the standalone harnesses compile the kernels with
exactly the shapes the production model uses.

Only ``block_topk`` is not spelled out in config.json: the model keeps a token
``budget`` and selects whole QSA blocks of ``r`` tokens, so ``block_topk`` defaults
to ``ceil(budget / r)``. Override it (and the sequence shape) from each test's CLI.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field

DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "..", "config.json")


@dataclass
class QSAConfig:
    # --- main attention ---
    heads_num: int          # Hq  (num_attention_heads)
    kv_heads_num: int       # Hkv (num_key_value_heads)
    k_head_size: int        # Dh  (head_dim)
    v_head_size: int        # Dv  (head_dim)
    hidden_size: int        # D   (hidden_size)
    # --- indexer ---
    indexer_heads: int      # Hidx (indexer_n_heads)
    indexer_kv_heads: int   # (indexer_kv_heads, always 1 here)
    indexer_head_dim: int   # Di  (indexer_head_dim)
    indexer_rotary_dim: int # rotary_dim = Di * partial_rotary_factor
    compress_ratio: int     # r   (indexer_compress_ratio)
    budget: int             # token budget (indexer_budget)
    block_topk: int         # blocks selected per query (derived: ceil(budget/r))
    # --- paged attention / misc ---
    pa_block_size: int      # PA page size (tokens per page)
    has_output_gate: bool   # output_gate_type == "sigmoid"
    is_causal: bool
    scale: float            # 1/sqrt(Dh)
    indexer_scale: float    # 1/sqrt(Di)
    indexer_norm_eps: float

    # derived JIT-only fields
    max_selected: int = field(default=0)

    @property
    def n_rep(self) -> int:
        return self.heads_num // self.kv_heads_num

    @property
    def summaries_per_page(self) -> int:
        return self.pa_block_size // self.compress_ratio

    def jit(self, kernel_name: str, **extra) -> dict:
        """The JIT #define set shared by every QSA kernel (mirrors make_qsa_jit)."""
        d = {
            "QSA_HEADS": self.heads_num,
            "QSA_KV_HEADS": self.kv_heads_num,
            "QSA_K_HEAD_SIZE": self.k_head_size,
            "QSA_V_HEAD_SIZE": self.v_head_size,
            "QSA_HIDDEN": self.hidden_size,
            "QSA_IDX_HEADS": self.indexer_heads,
            "QSA_IDX_HEAD_DIM": self.indexer_head_dim,
            "QSA_IDX_ROTARY_DIM": self.indexer_rotary_dim,
            "QSA_COMPRESS_RATIO": self.compress_ratio,
            "QSA_BLOCK_TOPK": self.block_topk,
            "QSA_BUDGET": self.budget,
            "QSA_MAX_SELECTED": self.max_selected or (self.block_topk * self.compress_ratio),
            "QSA_PA_BLOCK_SIZE": self.pa_block_size,
            "QSA_HAS_OUTPUT_GATE": 1 if self.has_output_gate else 0,
            "QSA_IS_CAUSAL": 1 if self.is_causal else 0,
            "QSA_SCALE": _fl(self.scale),
            "QSA_IDX_SCALE": _fl(self.indexer_scale),
            "QSA_IDX_NORM_EPS": _fl(self.indexer_norm_eps),
            "KERNEL_NAME": kernel_name,
        }
        d.update(extra)
        return d


def _fl(x: float) -> str:
    return f"{float(x):.8g}f"


def load(path: str | None = None, *, pa_block_size: int = 16,
         block_topk: int | None = None) -> QSAConfig:
    """Load QSA parameters from config.json.

    pa_block_size is the paged-attention page size; the OpenVINO default is 16
    (``cldnn::paged_attention::block_size``), which is what the production QSA
    tests use. block_topk overrides the ``ceil(budget/r)`` default.
    """
    path = path or DEFAULT_CONFIG
    with open(path) as fh:
        cfg = json.load(fh)
    t = cfg["text_config"]

    head_dim = int(t["head_dim"])
    di = int(t["indexer_head_dim"])
    prf = float(t.get("partial_rotary_factor", t.get("rope_parameters", {}).get("partial_rotary_factor", 1.0)))
    r = int(t["indexer_compress_ratio"])
    budget = int(t["indexer_budget"])
    topk = block_topk if block_topk is not None else math.ceil(budget / r)

    return QSAConfig(
        heads_num=int(t["num_attention_heads"]),
        kv_heads_num=int(t["num_key_value_heads"]),
        k_head_size=head_dim,
        v_head_size=head_dim,
        hidden_size=int(t["hidden_size"]),
        indexer_heads=int(t["indexer_n_heads"]),
        indexer_kv_heads=int(t.get("indexer_kv_heads", 1)),
        indexer_head_dim=di,
        indexer_rotary_dim=int(round(di * prf)),
        compress_ratio=r,
        budget=budget,
        block_topk=topk,
        pa_block_size=pa_block_size,
        has_output_gate=(t.get("output_gate_type", "sigmoid") == "sigmoid"),
        is_causal=True,
        scale=1.0 / math.sqrt(head_dim),
        indexer_scale=1.0 / math.sqrt(di),
        indexer_norm_eps=float(t.get("rms_norm_eps", 1e-6)),
    )


if __name__ == "__main__":
    c = load()
    import pprint
    pprint.pprint(c.__dict__)
    print("n_rep =", c.n_rep, "summaries_per_page =", c.summaries_per_page)
