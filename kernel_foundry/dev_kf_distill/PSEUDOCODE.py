"""
PSEUDOCODE.py — single-file, self-contained sketch of the distilled
LLM-driven, MAP-Elites-based GPU kernel optimizer described in this skill.

This file is the "concrete reference" the markdown docs link to. It is
intentionally one Python file with no imports outside the standard
library, so that anyone reading the skill can copy it and run it after
filling in just three things:

    1. llm_complete(prompt) -> str
    2. build(code)          -> (compiled: bool, log: str)
    3. run_correctness_test(code) -> (correct: bool, log: str, partial_perf_score: int)
    4. run_perf_test(code)  -> (runtime_ms: float, log: str)

Every other piece — descriptor classifier, archive, prompt pipeline,
evaluator, scoring, optional QD-gradient tracker, the main loop — is
included here in its minimum-viable form. Starter pattern dictionaries
and starter per-(dim, level, backend) instruction text are inlined so
the script runs end-to-end without external data.

For the algorithmic narrative behind each part of this file, read the
markdown docs in this same folder:

    SKILL.md                — top-level loop (you start here)
    BEHAVIOR_DESCRIPTOR.md  — classify_code, PATTERNS
    MAP_ELITES_LOOP.md      — Archive class
    PROMPT_PIPELINE.md      — build_prompt, INSTRUCTIONS, BASE_TEMPLATE
    EVALUATOR.md            — evaluate, combined_score
    QD_GRADIENT.md          — TransitionTracker

A full port for production use would replace the inlined PATTERNS and
INSTRUCTIONS tables with much larger backend-specific tables (~1500-2000
LoC of regex + ~400 LoC of JSON instructions), but the algorithm shape
in this file is complete.

Run it:

    $ python PSEUDOCODE.py --help            # NotImplementedError until wired
    $ python PSEUDOCODE.py --self-test       # exercises classifier, archive,
                                             # prompt builder; no LLM/build needed
"""

from __future__ import annotations

import argparse
import logging
import random
import re
import sys
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

# A behavioral coordinate in the 4D optimization space:
#   (memory_opt, compute_opt, parallelism_opt, esimd_opt) ∈ [0..3]^4
Coords = Tuple[int, int, int, int]


@dataclass
class Program:
    id: str
    code: str
    coords: Coords = (0, 0, 0, 0)
    score: float = 0.0
    runtime_ms: float = -1.0
    speedup: float = 0.0
    correct: bool = False
    compiled: bool = False
    parent_id: Optional[str] = None
    eval_log: str = ""


@dataclass
class EvalResult:
    perf_score: int = 0          # 0..5 ladder, see EVALUATOR.md §4.1
    compiled: bool = False
    correct: bool = False
    runtime_ms: float = -1.0
    speedup: float = 0.0
    eval_log: str = ""


# ---------------------------------------------------------------------------
# 1. BEHAVIOR DESCRIPTOR (BEHAVIOR_DESCRIPTOR.md)
# ---------------------------------------------------------------------------
#
# A starter pattern dictionary covering all four dimensions, with
# enough patterns for a small port to work. A production port would
# split this into per-backend overlays and add many more patterns;
# see BEHAVIOR_DESCRIPTOR.md §"How patterns are organized".
#
# Pattern structure for each dimension and level:
#     { level: { category: { "weight": float, "patterns": [regex, ...] }, ...}, ...}

PATTERNS: Dict[str, Dict[int, Dict[str, dict]]] = {
    # -- memory hierarchy exploitation --------------------------------------
    "memory": {
        0: {},
        1: {  # vectorized / coalesced global
            "vector_types": {
                "weight": 1.0,
                "patterns": [
                    r"\b(?:float|int|uint|half|double)[2348]\b",
                    r"sycl::vec<[^>]+,\s*[2348]>",
                    r"\bvload\d\b",
                    r"\bvstore\d\b",
                ],
            },
            "vector_load_store": {
                "weight": 1.0,
                "patterns": [
                    r"reinterpret_cast<[^>]*(?:float|int|double)[2348]\s*\*>",
                    r"as_(?:float|int|uint|double)[2348]\s*\(",
                    r"intel_sub_group_block_read\w*",
                    r"intel_sub_group_block_write\w*",
                ],
            },
            "aligned_access": {
                "weight": 0.7,
                "patterns": [
                    r"alignas\s*\(\s*\d+\s*\)",
                    r"__attribute__\s*\(\s*\(\s*aligned",
                    r"\b__ldg\b",
                ],
            },
            "restrict_qual": {
                "weight": 0.4,
                "patterns": [r"\b__restrict__\b", r"\brestrict\b"],
            },
        },
        2: {  # SLM / shared-memory tiling
            "local_decl": {
                "weight": 1.0,
                "patterns": [
                    r"__local\s+\w+",
                    r"__shared__\s+\w+",
                    r"local_accessor\s*<",
                    r"group_local_memory(?:_for_overwrite)?\s*<",
                ],
            },
            "barrier": {
                "weight": 0.7,
                "patterns": [
                    r"barrier\s*\(\s*CLK_LOCAL_MEM_FENCE",
                    r"__syncthreads\s*\(\s*\)",
                    r"sycl::group_barrier",
                    r"item\.barrier\s*\(",
                ],
            },
            "tile_naming": {
                "weight": 0.6,
                "patterns": [
                    r"\b(?:tile|smem|slm|shared_mem)_?\w*\s*\[",
                    r"\bTILE_[MNK]\b",
                    r"\bBLOCK_[MNK]\b",
                ],
            },
        },
        3: {  # register blocking + async / multi-stage
            "async_copy": {
                "weight": 1.0,
                "patterns": [
                    r"async_work_group_copy",
                    r"cuda::memcpy_async",
                    r"cp\.async\b",
                ],
            },
            "double_buffer": {
                "weight": 0.9,
                "patterns": [
                    r"\b\w+\s*\[\s*2\s*\]\s*\[",      # foo[2][...]
                    r"\bbuf\s*\^=\s*1\b",
                    r"\bping\b.*\bpong\b",
                ],
            },
            "register_tile": {
                "weight": 0.8,
                "patterns": [
                    r"#pragma\s+unroll",
                    r"\bfmaf?\s*\(",
                    r"\breg_[ABCD]\s*\[",
                ],
            },
            "tma_or_2d_block": {
                "weight": 1.0,
                "patterns": [
                    r"intel_sub_group_2d_block_read",
                    r"\bcp\.async\.bulk\b",
                    r"\bTMA\b",
                ],
            },
        },
    },

    # -- algorithmic / compute ----------------------------------------------
    "compute": {
        0: {},
        1: {  # fused operations
            "fused_arith": {
                "weight": 0.7,
                "patterns": [r"\bfma[fd]?\s*\(", r"\bmad\s*\(", r"#pragma\s+unroll"],
            },
            "single_kernel_fuse": {
                "weight": 0.6,
                "patterns": [
                    r"//\s*fused",
                    r"\b__forceinline__\b",
                    r"\binline\s+\w+\s+\w+\s*\(",
                ],
            },
        },
        2: {  # streaming / online algorithm
            "online_softmax": {
                "weight": 0.9,
                "patterns": [
                    r"\brunning_max\b",
                    r"\brunning_sum\b",
                    r"online[_-]?softmax",
                    r"\bmax_val\b.*?\bsum_val\b",
                ],
            },
            "welford": {
                "weight": 0.9,
                "patterns": [r"\bM2\b.*?delta", r"welford"],
            },
            "kahan": {
                "weight": 0.8,
                "patterns": [r"\bcompensation\b", r"kahan"],
            },
        },
        3: {  # tiled / blocked
            "tiled_loop": {
                "weight": 0.9,
                "patterns": [
                    r"for\s*\(\s*int\s+\w+\s*=\s*0\s*;[^;]*;\s*\w+\s*\+=\s*TILE",
                    r"for\s*\(\s*int\s+k_tile\s*=",
                    r"\bBLOCK_K\b.*\bTILE_M\b",
                ],
            },
            "flash_attn_pattern": {
                "weight": 1.0,
                "patterns": [
                    r"flash[_-]attention",
                    r"row_max.*row_sum.*row_output",
                ],
            },
        },
    },

    # -- parallelism granularity --------------------------------------------
    "parallelism": {
        0: {},
        1: {  # work-group barriers / collective
            "wg_barrier": {
                "weight": 0.7,
                "patterns": [
                    r"barrier\s*\(\s*CLK_LOCAL_MEM_FENCE",
                    r"__syncthreads\s*\(\s*\)",
                    r"sycl::group_barrier",
                ],
            },
            "wg_reduce": {
                "weight": 0.8,
                "patterns": [
                    r"work_group_reduce_\w+",
                    r"reduce_over_group\s*\(\s*item\.get_group",
                    r"BlockReduce\s*<",
                ],
            },
        },
        2: {  # sub-group / warp
            "sg_reduce": {
                "weight": 1.0,
                "patterns": [
                    r"sub_group_reduce_\w+",
                    r"reduce_over_group\s*\(\s*sg",
                    r"__shfl_(?:xor_)?sync",
                    r"__shfl_down_sync",
                    r"warp_reduce_\w+",
                ],
            },
            "sg_shuffle": {
                "weight": 0.9,
                "patterns": [
                    r"shift_group_(?:left|right)",
                    r"group_broadcast\s*\(\s*sg",
                    r"sub_group_broadcast",
                    r"__shfl_sync\b",
                ],
            },
        },
        3: {  # hierarchical (thread + warp + block)
            "hierarchical_decomp": {
                "weight": 0.9,
                "patterns": [
                    r"\bSUBTILE_[MN]\b",
                    r"\bTHREAD_[MN]\b",
                    r"// Level 1.*// Level 2.*// Level 3",
                ],
            },
            "wg_plus_sg": {
                "weight": 0.7,
                "patterns": [
                    # work-group barrier AND sub-group reduce in same file
                    r"(?s)work_group_reduce.*sub_group_reduce",
                    r"(?s)__syncthreads.*__shfl",
                ],
            },
        },
    },

    # -- explicit SIMD / tensor-core / DPAS --------------------------------
    "esimd": {
        0: {},
        1: {  # basic
            "esimd_basic": {
                "weight": 0.9,
                "patterns": [
                    r"sycl::ext::intel::esimd",
                    r"\besimd::\w+",
                    r"#include\s*<sycl/ext/intel/esimd",
                ],
            },
            "wmma_basic": {
                "weight": 0.9,
                "patterns": [r"\bwmma::\w+", r"#include\s*<mma\.h>"],
            },
            "ocl_dpas_basic": {
                "weight": 0.9,
                "patterns": [r"intel_sub_group_dpas\w*"],
            },
        },
        2: {  # optimized
            "simd_typed": {
                "weight": 0.9,
                "patterns": [
                    r"\bsimd<\s*\w+\s*,\s*\d+\s*>",
                    r"esimd::block_load\s*<",
                    r"esimd::block_store\s*<",
                ],
            },
            "mma_sync": {
                "weight": 0.9,
                "patterns": [r"\bmma_sync\b", r"\bmma\.sync\b"],
            },
        },
        3: {  # expert
            "expert_simd": {
                "weight": 1.0,
                "patterns": [
                    r"intel_sub_group_2d_block_read",
                    r"lsc_block_load\s*<",
                    r"cp\.async\.bulk\.tensor",
                    r"\bTMA\b",
                ],
            },
            "grouped_mma": {
                "weight": 1.0,
                "patterns": [
                    r"grouped_gemm",
                    r"cute::tiled_mma",
                ],
            },
        },
    },
}


def _classify_dimension(code: str, dim_patterns: Dict[int, Dict[str, dict]]) -> int:
    """Highest level whose weighted-pattern score crosses the threshold.

    See BEHAVIOR_DESCRIPTOR.md §"How classification works":
        - level 1 threshold 0.15
        - level 2 threshold 0.18
        - level 3 threshold 0.21
        - definitive single category (weight ≥ 0.9) floors score at 0.25
    """
    for level in (3, 2, 1):
        cats = dim_patterns.get(level, {})
        if not cats:
            continue
        total_w = matched_w = max_single = 0.0
        for _, cfg in cats.items():
            w = cfg.get("weight", 1.0)
            total_w += w
            for pat in cfg.get("patterns", []):
                try:
                    if re.search(pat, code, flags=re.IGNORECASE | re.MULTILINE):
                        matched_w += w
                        max_single = max(max_single, w)
                        break
                except re.error:
                    continue  # malformed pattern — skip
        if total_w == 0:
            continue
        score = matched_w / total_w
        if max_single >= 0.9 and matched_w > 0:
            score = max(score, 0.25)
        threshold = 0.15 + (level - 1) * 0.03
        if score >= threshold:
            return level
    return 0


def classify_code(code: str) -> Coords:
    """Static, deterministic, language-agnostic-here classification.

    A production port branches on `language` and uses backend-specific
    pattern overlays — see BEHAVIOR_DESCRIPTOR.md §"How patterns are
    organized".
    """
    if not code:
        return (0, 0, 0, 0)
    return (
        _classify_dimension(code, PATTERNS["memory"]),
        _classify_dimension(code, PATTERNS["compute"]),
        _classify_dimension(code, PATTERNS["parallelism"]),
        _classify_dimension(code, PATTERNS["esimd"]),
    )


# ---------------------------------------------------------------------------
# 2. ARCHIVE (MAP_ELITES_LOOP.md)
# ---------------------------------------------------------------------------

class Archive:
    """A 4×4×4×4 MAP-Elites grid with optional islands.

    Each cell holds at most one elite — the highest-scoring candidate
    that has ever landed there. See MAP_ELITES_LOOP.md for the full
    design.
    """

    def __init__(self, num_islands: int = 1):
        self.feature_map: Dict[Coords, str] = {}
        self.programs: Dict[str, Program] = {}
        self.num_islands = max(1, num_islands)
        self.islands: List[set] = [set() for _ in range(self.num_islands)]
        self.current_island = 0
        self._island_counter = 0

    def is_empty(self) -> bool:
        return not self.feature_map

    def add(self, program: Program) -> bool:
        """Returns True iff `program` became the elite of its cell."""
        program.coords = classify_code(program.code)
        self.programs[program.id] = program
        self.islands[self.current_island].add(program.id)

        cur_id = self.feature_map.get(program.coords)
        if cur_id is None or program.score > self.programs[cur_id].score:
            self.feature_map[program.coords] = program.id
            return True
        return False

    def underexplored_cells(self) -> List[Coords]:
        all_cells = [(m, c, p, e)
                     for m in range(4) for c in range(4)
                     for p in range(4) for e in range(4)]
        empty = [k for k in all_cells if k not in self.feature_map]
        if empty:
            return empty
        scores = sorted(self.programs[pid].score
                        for pid in self.feature_map.values())
        if not scores:
            return all_cells
        median = scores[len(scores) // 2]
        return [k for k, pid in self.feature_map.items()
                if self.programs[pid].score < median]

    def island_elites(self, island: int) -> List[Program]:
        ids = self.islands[island] & set(self.feature_map.values())
        return [self.programs[pid] for pid in ids]

    def sample(self, exploit: float = 0.7, explore: float = 0.2) -> Optional[Program]:
        """Three-way mix: exploit / explore / random.

        See MAP_ELITES_LOOP.md §2.3.
        """
        elites = self.island_elites(self.current_island)
        if not elites:
            elites = [self.programs[pid] for pid in self.feature_map.values()]
        if not elites:
            return None

        r = random.random()
        if r < exploit:
            elites_sorted = sorted(elites, key=lambda p: -p.score)
            top_k = elites_sorted[:max(1, len(elites_sorted) // 3)]
            return random.choice(top_k)
        if r < exploit + explore:
            cells = self.underexplored_cells()
            # exploration mode: still pick a parent (the cells guide
            # target-profile selection elsewhere; here we just diversify)
            return random.choice(elites)
        return random.choice(elites)

    def top_k_in_island(self, island: int, k: int = 3) -> List[Program]:
        elites = sorted(self.island_elites(island), key=lambda p: -p.score)
        return elites[:k]

    def best(self) -> Optional[Program]:
        if not self.feature_map:
            return None
        return max((self.programs[pid] for pid in self.feature_map.values()),
                   key=lambda p: p.score)

    def advance_island_counter(self, programs_per_island: int = 5) -> None:
        self._island_counter += 1
        if self._island_counter >= programs_per_island:
            self._island_counter = 0
            self.current_island = (self.current_island + 1) % self.num_islands

    def migrate_islands(self, migration_rate: float = 0.1) -> None:
        """Ring migration: copy top fraction of each island to the next."""
        if self.num_islands < 2:
            return
        for src in range(self.num_islands):
            dst = (src + 1) % self.num_islands
            elites = sorted(self.island_elites(src), key=lambda p: -p.score)
            n = max(1, int(len(elites) * migration_rate))
            for prog in elites[:n]:
                self.islands[dst].add(prog.id)


# ---------------------------------------------------------------------------
# 3. PROMPT PIPELINE (PROMPT_PIPELINE.md)
# ---------------------------------------------------------------------------
#
# Inlined per-(dimension, backend, level) instruction text. Real
# deployments use ~400 LoC of JSON; this is enough for a working
# port. Keys with no entry produce no text (e.g. memory level 0).

INSTRUCTIONS: Dict[Tuple[str, str, int], str] = {
    # ---- memory ----
    ("memory", "ocl", 1): (
        "**Vectorized & Coalesced Global Memory**: replace scalar loads with "
        "`vload4` / `vstore4` (or `intel_sub_group_block_read_us` for "
        "sub-group-coalesced loads). Ensure adjacent work-items access "
        "adjacent memory. Add `restrict` to non-aliasing pointers."
    ),
    ("memory", "ocl", 2): (
        "**Local Memory Tiling**: stage a TILE_H × (TILE_W+1) block of input "
        "in `__local`, with `barrier(CLK_LOCAL_MEM_FENCE)` before reads. "
        "Add the +1 column padding to avoid bank conflicts on column-wise access."
    ),
    ("memory", "ocl", 3): (
        "**Register Blocking + Double-Buffer**: each work-item accumulates a "
        "THREAD_M × THREAD_N register tile via `#pragma unroll` inner loops; "
        "use a double-buffered SLM (`tile[2][...]`) so the next K-tile is "
        "loaded while the current one is consumed."
    ),
    ("memory", "sycl", 1): (
        "**Vectorized & Coalesced Global Memory**: use `sycl::vec<T,4>` or "
        "`reinterpret_cast<sycl::float4*>` for wider transactions. Ensure "
        "adjacent items access adjacent addresses; add `[[intel::aligned(16)]]`."
    ),
    ("memory", "sycl", 2): (
        "**SLM Tiling**: allocate `group_local_memory_for_overwrite<float[TILE][TILE+1]>` "
        "with a +1 padding column to avoid bank conflicts. Use "
        "`sycl::group_barrier(item.get_group())` between writes and reads."
    ),
    ("memory", "sycl", 3): (
        "**Register Blocking with SLM Double-Buffer**: maintain a private "
        "`reg_C[THREAD_M][THREAD_N]` accumulator per work-item, with all inner "
        "loops `#pragma unroll`'d. Pipeline the next K-tile load while the "
        "current one is consumed."
    ),
    ("memory", "cuda", 1): (
        "**Vectorized & Coalesced Global Memory**: use `float4` / `int4` via "
        "`reinterpret_cast`. Use `__ldg(&x[i])` for read-only data. Annotate "
        "non-aliasing pointers with `__restrict__`."
    ),
    ("memory", "cuda", 2): (
        "**Shared Memory Tiling**: declare `__shared__ float tile[TILE][TILE+1]` "
        "(padded), cooperatively load with all threads, and call "
        "`__syncthreads()` between writes and reads."
    ),
    ("memory", "cuda", 3): (
        "**Register Blocking + cp.async Double-Buffer**: each thread keeps a "
        "`reg_C[THREAD_M][THREAD_N]` register tile (e.g. 8×8). Use "
        "`cuda::memcpy_async` (or `cp.async`) to overlap the next tile's "
        "global→shared transfer with current-tile compute."
    ),

    # ---- compute ----
    ("compute", "ocl", 1): (
        "**Kernel Fusion**: collapse sequential element-wise stages into one "
        "kernel; keep intermediates in registers. Use `mad(a,b,c)` / "
        "`fma(a,b,c)` for multiply-add chains."
    ),
    ("compute", "ocl", 2): (
        "**Single-Pass / Online**: replace two-pass max-then-sum reductions "
        "with a running-(max, sum) update; for variance, use Welford's "
        "online algorithm."
    ),
    ("compute", "ocl", 3): (
        "**Tiled / Blocked Algorithm**: process inputs in `BLOCK_K` chunks, "
        "maintain running statistics across chunks, and trade selective "
        "recomputation for lower intermediate footprint."
    ),
    ("compute", "sycl", 1): (
        "**Kernel Fusion**: combine sequential element-wise ops into one "
        "kernel; use `sycl::fma` / `sycl::mad` to keep intermediates in "
        "registers."
    ),
    ("compute", "sycl", 2): (
        "**Online Softmax / Welford**: maintain running (max, sum) with "
        "rescale-on-update so a single pass produces both stages; use "
        "Welford for mean+variance in one pass."
    ),
    ("compute", "sycl", 3): (
        "**Flash-Attention-Style Blocked**: process input in BLOCK_SIZE=64 "
        "chunks, maintain `row_max` / `row_sum` across blocks with "
        "rescale-on-update, and recompute exp(x-max) on the output pass "
        "instead of storing it."
    ),
    ("compute", "cuda", 1): (
        "**Kernel Fusion + Fast Math**: fuse element-wise stages; use "
        "`fmaf`, `__expf`, `__logf`, `__fdividef`, `rsqrtf` for tight inner loops."
    ),
    ("compute", "cuda", 2): (
        "**Online Softmax with Warp Shuffle**: maintain running (max, sum) "
        "and merge across the warp via `__shfl_xor_sync` — no shared memory "
        "needed for the reduction."
    ),
    ("compute", "cuda", 3): (
        "**Flash-Attention + WMMA**: process Q in blocks of 64, recompute "
        "exp(qk-max) on the output pass, and use `wmma::fragment` + "
        "`mma_sync` for the matmul portion. CUB block-level primitives for "
        "any non-MMA reductions."
    ),

    # ---- parallelism ----
    ("parallelism", "ocl", 1): (
        "**Work-Group Tree Reduction**: replace atomic accumulation with a "
        "log₂(WG_SIZE) tree reduction in `__local` memory, with "
        "`barrier(CLK_LOCAL_MEM_FENCE)` between each stride."
    ),
    ("parallelism", "ocl", 2): (
        "**Sub-Group Collectives**: use `sub_group_reduce_add`, "
        "`sub_group_broadcast`, and `intel_sub_group_block_read_us` instead "
        "of SLM-based reductions. No barrier needed within a sub-group."
    ),
    ("parallelism", "ocl", 3): (
        "**Hierarchical (WG → SG → Lane)**: structure indices as "
        "`(group_id, sg_id, lane)` and let each level cache at the "
        "appropriate granularity (SLM at WG, shuffles at SG, registers at "
        "lane)."
    ),
    ("parallelism", "sycl", 1): (
        "**Tree Reduction in SLM**: do a log₂(WG_SIZE) reduction in "
        "`group_local_memory`, with `sycl::group_barrier` between strides."
    ),
    ("parallelism", "sycl", 2): (
        "**Sub-Group Collectives**: use `sycl::reduce_over_group(sg, x, sycl::plus<float>())`, "
        "`sycl::group_broadcast`, `sycl::shift_group_left/right`. NEVER pass "
        "a custom lambda to `reduce_over_group` — only the predefined "
        "function objects work."
    ),
    ("parallelism", "sycl", 3): (
        "**Three-Level Decomposition**: TILE_M×TILE_N at the work-group, "
        "SUBTILE_M×SUBTILE_N at the sub-group, THREAD_M×THREAD_N per "
        "work-item. Sub-group shuffles replace SLM at the SG level."
    ),
    ("parallelism", "cuda", 1): (
        "**Block-level Reduction**: use `cub::BlockReduce<float, BLOCK>::Sum` "
        "or a hand-rolled `__shared__` tree-reduction with `__syncthreads`."
    ),
    ("parallelism", "cuda", 2): (
        "**Warp Shuffle Reduction**: replace block-level reduce with "
        "`__shfl_xor_sync(0xffffffff, val, offset)` for offset=16,8,4,2,1. "
        "No shared memory needed, no barriers."
    ),
    ("parallelism", "cuda", 3): (
        "**Hierarchical (Block → Warp → Thread)**: lay out work as "
        "(blockIdx, warpId, laneId); accumulate per-thread register tiles, "
        "merge across warps via shuffle, merge across the block via shared."
    ),

    # ---- esimd / tensor-core / DPAS ----
    ("esimd", "ocl", 1): (
        "**Sub-Group DPAS (basic)**: emit `intel_sub_group_dpas` with 8x16 "
        "block accumulators for the inner matmul. Requires sub-group size 16."
    ),
    ("esimd", "ocl", 2): (
        "**DPAS + Block IO**: combine `intel_sub_group_dpas` with "
        "`intel_sub_group_block_read_us` for B-loads and `block_write` for "
        "C-stores. Use `restrict` on all pointers."
    ),
    ("esimd", "ocl", 3): (
        "**DPAS + 2D Block Read**: use `intel_sub_group_2d_block_read` for "
        "tiled A/B loads (avoids the 1D coalescing constraint), feed "
        "directly into `intel_sub_group_dpas` accumulators."
    ),
    ("esimd", "sycl", 1): (
        "**Basic ESIMD**: include `<sycl/ext/intel/esimd.hpp>`; use "
        "`esimd::simd<float, N>` for explicit-SIMD vectors and "
        "`esimd::block_load` / `block_store` for memory ops."
    ),
    ("esimd", "sycl", 2): (
        "**Cooperative ESIMD**: organize the kernel around `simd<T, N>` "
        "fragments matching the hardware's SIMD width; use `esimd::dpas` "
        "for matrix multiply-accumulate."
    ),
    ("esimd", "sycl", 3): (
        "**Expert ESIMD**: `lsc_block_load` with cache hints + `esimd::dpas` "
        "8×8×16 grouped MMA + 2D block read where available. Hand-tune "
        "register layout to match hardware GRF banks."
    ),
    ("esimd", "cuda", 1): (
        "**Basic WMMA**: include `<mma.h>`; use 16×16×16 `wmma::fragment` "
        "for matrix-A, matrix-B, accumulator; `load_matrix_sync` + "
        "`mma_sync` + `store_matrix_sync`."
    ),
    ("esimd", "cuda", 2): (
        "**mma.sync Inline PTX**: drop to `mma.sync.aligned.m16n8k16.row.col` "
        "for fine-grained control; pack accumulators in registers across "
        "K-iterations."
    ),
    ("esimd", "cuda", 3): (
        "**Async-copy + Grouped MMA + TMA**: `cp.async.bulk.tensor` for "
        "A/B loads, `cute::tiled_mma` (CUTLASS-style) for the inner MMA, "
        "double-buffered shared memory pipeline. Hopper+ only."
    ),
}

# Layered base template — see PROMPT_PIPELINE.md §3.1.
BASE_TEMPLATE = """You are an expert {backend} engineer specializing in GPU kernel optimization.
Given a reference implementation, write a performant {backend} kernel with identical functionality.

## Reference
```
{reference}
```

## Hardware
{hardware}

## Task
{task_objectives}

## Critical Requirements
1. The kernel must exactly match the reference's functionality.
2. The code must compile and run on the GPU.
3. Keep input shapes / hyperparameters unchanged from the reference.
4. Provide the complete kernel as a single code block.
"""


def render_optimization_instructions(target: Coords, backend: str) -> str:
    """See PROMPT_PIPELINE.md §3.4."""
    parts: List[str] = []
    for dim, level in zip(("memory", "compute", "parallelism", "esimd"), target):
        if level >= 1:
            txt = INSTRUCTIONS.get((dim, backend, level))
            if txt:
                parts.append(f"- {txt}")
    if not parts:
        return ""
    return ("\n## Required Optimizations\n\n"
            "Apply the following techniques in your implementation:\n\n"
            + "\n\n".join(parts) + "\n")


def pick_target_profile(parent: Optional[Program], strategy: str,
                        underexplored: List[Coords]) -> Optional[Coords]:
    """See PROMPT_PIPELINE.md §3.5."""
    if parent is None or not parent.code:
        return random.choice(underexplored) if underexplored else None
    pm, pc, pp, pe = parent.coords

    if strategy == "diversify":
        return random.choice(underexplored) if underexplored else parent.coords

    if strategy == "esimd_upgrade" and pe == 0:
        return (pm, pc, pp, random.randint(1, 3))

    # default: mutate — 1-step neighbor in a randomly chosen dimension,
    # weighted slightly toward unmaxed dimensions.
    target = list(parent.coords)
    candidate_dims = [i for i, v in enumerate(target) if v < 3] or [0, 1, 2, 3]
    dim = random.choice(candidate_dims)
    delta = random.choice([-1, +1, +1])    # bias toward escalation
    target[dim] = max(0, min(3, target[dim] + delta))
    return tuple(target)


def build_prompt(reference_code: str, parent: Optional[Program],
                 inspirations: List[Program], target: Optional[Coords],
                 backend: str, hardware: str,
                 recent_compile_errors: List[str],
                 mutation_hints: Optional[List[str]] = None) -> str:
    """Layered prompt — see PROMPT_PIPELINE.md §3 for the full shape."""
    # Mode selection (status="error" / "correct" / "translate").
    if parent is None or not parent.code:
        status = "translate"
        objectives = (
            "1. Analyze the requirements for an efficient {b} kernel.\n"
            "2. Explain your kernel structure step by step.\n"
            "3. Provide the complete {b} code in a code block."
        ).format(b=backend)
    elif not parent.compiled or not parent.correct:
        status = "error"
        objectives = (
            "1. Identify the errors in the previous kernel from the eval log.\n"
            "2. Explain your fixes step by step.\n"
            "3. Provide the complete, corrected code in a code block."
        )
    else:
        status = "correct"
        objectives = (
            "1. Analyze why the previous kernel achieved its current speed.\n"
            "2. Identify the dominant bottleneck (memory / compute / parallelism).\n"
            "3. Apply the Required Optimizations below.\n"
            "4. Provide the complete, improved kernel in a code block."
        )

    parts: List[str] = []
    parts.append(BASE_TEMPLATE.format(
        backend=backend, reference=reference_code,
        hardware=hardware, task_objectives=objectives,
    ))

    # 3.3 Evolution context.
    if parent and parent.code:
        parts.append(
            f"## Previous Version "
            f"(score={parent.score:.2f}, runtime={parent.runtime_ms:.2f} ms, "
            f"correct={parent.correct}, compiled={parent.compiled})\n"
            f"```\n{parent.code}\n```"
        )
        if parent.eval_log:
            tail = parent.eval_log[-2000:]
            parts.append(f"### Console output (tail):\n```\n{tail}\n```")
    for i, ins in enumerate(inspirations[:3]):
        parts.append(
            f"## Inspiration {i+1} (score={ins.score:.2f}, "
            f"runtime={ins.runtime_ms:.2f} ms)\n```\n{ins.code}\n```"
        )

    # 3.4 Target-profile instructions.
    if target and status == "correct":
        instructions = render_optimization_instructions(target, backend)
        if instructions:
            parts.append(instructions)

    # 3.6 Gradient mutation hints.
    if mutation_hints:
        bullets = "\n".join(f"- {h}" for h in mutation_hints[:3])
        parts.append(
            "## Mutation Hints (from history)\n"
            "Transitions that historically improved fitness from the "
            "current cell:\n" + bullets
        )

    # 3.7 Compile-error feedback.
    if recent_compile_errors:
        bullets = "\n".join(f"- Branch {i}: {err[:500]}"
                            for i, err in enumerate(recent_compile_errors[:3]))
        parts.append(
            "## Avoid these compile errors from prior trials\n" + bullets
        )

    return "\n\n".join(parts)


def extract_code(llm_response: str) -> str:
    """Pull the largest fenced code block out of an LLM response.

    Falls back to the whole response if no fenced block is found.
    """
    if not llm_response:
        return ""
    blocks = re.findall(r"```[\w+-]*\n(.*?)```", llm_response, flags=re.DOTALL)
    if not blocks:
        return llm_response.strip()
    return max(blocks, key=len).strip()


# ---------------------------------------------------------------------------
# 4. EVALUATOR (EVALUATOR.md)
# ---------------------------------------------------------------------------

def evaluate(code: str, reference_runtime_ms: float) -> EvalResult:
    """Five-tier ladder. See EVALUATOR.md §4.1.

    Wire `build`, `run_correctness_test`, `run_perf_test` to your
    actual harness before calling.
    """
    # T0: extract — if there is no plausible code body, return 0.
    if not code or len(code.strip()) < 16:
        return EvalResult(perf_score=0, eval_log="extraction failed (no code body)")

    # T0.5 / T1: compile.
    compiled, compile_log = build(code)
    if not compiled:
        return EvalResult(perf_score=1, compiled=False, eval_log=compile_log)

    # T2: correctness.
    correct, correctness_log, partial_score = run_correctness_test(code)
    if not correct:
        return EvalResult(
            perf_score=partial_score, compiled=True, correct=False,
            eval_log=compile_log + "\n" + correctness_log,
        )

    # T3: performance.
    runtime_ms, perf_log = run_perf_test(code)
    speedup = (reference_runtime_ms / runtime_ms) if runtime_ms > 0 else 0.0
    return EvalResult(
        perf_score=5, compiled=True, correct=True,
        runtime_ms=runtime_ms, speedup=speedup,
        eval_log=(compile_log + "\n" + correctness_log + "\n" + perf_log),
    )


def combined_score(r: EvalResult) -> float:
    """See EVALUATOR.md §4.2."""
    return r.perf_score + 3 * int(r.correct and r.speedup > 0) * r.speedup


def select_best_solution(results: List[EvalResult]) -> int:
    """See EVALUATOR.md §4.3."""
    runtimes = [r.runtime_ms if r.runtime_ms > 0 else float("inf") for r in results]
    if all(rt == float("inf") for rt in runtimes):
        scores = [r.perf_score for r in results]
        return max(range(len(results)), key=lambda i: scores[i])
    return min(range(len(results)), key=lambda i: runtimes[i])


# Wiring stubs — fill these in for your environment ----------------------
def build(code: str) -> Tuple[bool, str]:
    """Compile the candidate. Return (compiled, log)."""
    raise NotImplementedError(
        "Plug in your build harness (e.g. icpx / nvcc / clBuildProgram). "
        "Return (True, log) on success and (False, error_log) on failure."
    )


def run_correctness_test(code: str) -> Tuple[bool, str, int]:
    """Run correctness. Return (correct, log, partial_perf_score).

    `partial_perf_score`: 2 if runtime crash, 3 if shape mismatch,
    4 if value mismatch, 5 if correct (the caller bumps to 5 itself).
    """
    raise NotImplementedError("Plug in your correctness harness.")


def run_perf_test(code: str) -> Tuple[float, str]:
    """Run timed performance. Return (runtime_ms, log)."""
    raise NotImplementedError("Plug in your performance harness.")


def llm_complete(prompt: str) -> str:
    """Call the LLM. Return its raw text response."""
    raise NotImplementedError(
        "Plug in your LLM client (e.g. anthropic, openai). Pass `prompt` as "
        "a single user message; return the assistant's full text."
    )


# ---------------------------------------------------------------------------
# 5. QD GRADIENT TRACKER (QD_GRADIENT.md) — minimum viable version
# ---------------------------------------------------------------------------

class TransitionTracker:
    """Empirical-frequency gradient over (parent_coords → child_coords)
    transitions. See QD_GRADIENT.md.

    This MVP keeps a circular buffer of recent transitions and exposes:
        - record(parent, child, became_elite)
        - mutation_hints_for(parent, top_k=3)

    A production version adds an LRU per-cell cache of statistics, a
    softmax sampling-weight function, and a multi-component (fitness +
    improvement-rate + exploration) gradient. None of that is needed to
    get useful behavior; this MVP is enough.
    """

    DIM_NAMES = ("memory_opt", "compute_opt", "parallelism_opt", "esimd_opt")

    def __init__(self, max_history: int = 10000):
        self.records: deque = deque(maxlen=max_history)

    def record(self, parent: Optional[Program], child: Program,
               became_elite: bool) -> None:
        if parent is None:
            return
        delta_score = child.score - parent.score
        improved = (delta_score > 0) or became_elite
        self.records.append({
            "parent": parent.coords,
            "child": child.coords,
            "improved": improved,
        })

    def mutation_hints_for(self, parent: Program, top_k: int = 3) -> List[str]:
        """Aggregate by transition direction *from this cell*.

        Returns up to `top_k` human-readable hints for directions whose
        historical success rate exceeds 50% with at least 2 samples.
        """
        if not parent or not self.records:
            return []
        # Aggregate (delta_vector → counts) for transitions starting at
        # parent.coords (or any neighboring cell, to grow the sample set).
        from_here: Dict[Tuple[int, int, int, int], List[bool]] = {}
        for rec in self.records:
            if rec["parent"] != parent.coords:
                continue
            d = tuple(c - p for c, p in zip(rec["child"], rec["parent"]))
            from_here.setdefault(d, []).append(rec["improved"])

        ranked: List[Tuple[Tuple[int, int, int, int], float, int]] = []
        for direction, outcomes in from_here.items():
            if len(outcomes) < 2 or direction == (0, 0, 0, 0):
                continue
            success = sum(outcomes) / len(outcomes)
            if success > 0.5:
                ranked.append((direction, success, len(outcomes)))
        ranked.sort(key=lambda r: (-r[1], -r[2]))

        out: List[str] = []
        for direction, rate, n in ranked[:top_k]:
            changes = []
            for dim_name, delta in zip(self.DIM_NAMES, direction):
                if delta != 0:
                    word = "increase" if delta > 0 else "decrease"
                    changes.append(f"{word} {dim_name}")
            if not changes:
                continue
            out.append(
                f"{' and '.join(changes)} — historical success rate "
                f"{int(rate * 100)}% over {n} trials"
            )
        return out


# ---------------------------------------------------------------------------
# 6. THE MAIN LOOP (SKILL.md §pseudocode)
# ---------------------------------------------------------------------------

def run(reference_code: str, reference_runtime_ms: float, *,
        backend: str = "ocl",
        hardware_description: str = "Generic GPU.",
        max_iters: int = 10,
        branches_per_iteration: int = 4,
        exploration_strategy: str = "mutate",
        early_stopping_patience: int = 2,
        early_stopping_min_trials: int = 3,
        num_islands: int = 1,
        migration_interval: int = 10,
        use_gradient_tracking: bool = True,
        seed: Optional[int] = None) -> Optional[Program]:
    """Drive the whole loop. Returns the archive best, or None.

    Wire `llm_complete`, `build`, `run_correctness_test`, `run_perf_test`
    above before calling.
    """
    if seed is not None:
        random.seed(seed)

    archive = Archive(num_islands=num_islands)
    tracker = TransitionTracker() if use_gradient_tracking else None

    # Seed with the reference itself as program0.
    program0 = Program(
        id="prog-0", code=reference_code,
        coords=classify_code(reference_code),
        score=5.0,                  # 5 = correct, no speedup multiplier
        runtime_ms=reference_runtime_ms,
        speedup=1.0, correct=True, compiled=True,
        eval_log="(reference baseline — not LLM-generated)",
    )
    archive.add(program0)

    recent_compile_errors: List[str] = []
    best_runtime = reference_runtime_ms
    stagnation = 0

    for trial in range(max_iters):
        branches: List[Program] = []
        trial_compile_errors: List[str] = []

        for branch_idx in range(branches_per_iteration):
            # 1. SAMPLE PARENT
            parent = archive.sample() if not archive.is_empty() else program0

            # 2. SELECT TARGET PROFILE
            target = pick_target_profile(
                parent, exploration_strategy,
                archive.underexplored_cells(),
            )

            # Inspirations: top-3 elites in current island (excluding parent).
            inspirations = [
                p for p in archive.top_k_in_island(archive.current_island, k=4)
                if p.id != parent.id
            ][:3]

            # 3. BUILD PROMPT
            hints = tracker.mutation_hints_for(parent, top_k=3) if tracker else None
            prompt = build_prompt(
                reference_code, parent, inspirations, target,
                backend, hardware_description, recent_compile_errors, hints,
            )

            # 4. LLM
            llm_response = llm_complete(prompt)
            code = extract_code(llm_response)

            # 5. EVALUATE
            result = evaluate(code, reference_runtime_ms)

            # 6. SCORE + UPDATE
            score = combined_score(result)
            child = Program(
                id=f"prog-{trial}-{branch_idx}",
                code=code,
                coords=classify_code(code),
                score=score,
                runtime_ms=result.runtime_ms,
                speedup=result.speedup,
                correct=result.correct,
                compiled=result.compiled,
                parent_id=parent.id,
                eval_log=result.eval_log,
            )
            became_elite = archive.add(child)
            archive.advance_island_counter()
            if tracker:
                tracker.record(parent, child, became_elite)
            branches.append(child)

            if not result.compiled and result.eval_log:
                trial_compile_errors.append(result.eval_log[:500])

            logging.info(
                "trial=%d branch=%d coords=%s score=%.2f speedup=%.2fx elite=%s",
                trial, branch_idx, child.coords, child.score,
                child.speedup, became_elite,
            )

        recent_compile_errors = (recent_compile_errors + trial_compile_errors)[-3:]

        # 7. EARLY STOP / MIGRATE
        trial_best = min((b.runtime_ms for b in branches if b.runtime_ms > 0),
                        default=float("inf"))
        if trial_best < best_runtime:
            best_runtime = trial_best
            stagnation = 0
        else:
            stagnation += 1

        if (trial >= early_stopping_min_trials
                and stagnation >= early_stopping_patience):
            logging.info("early-stopping at trial=%d, best=%.3f ms", trial, best_runtime)
            break

        if migration_interval and (trial + 1) % migration_interval == 0:
            archive.migrate_islands()

    return archive.best()


# ---------------------------------------------------------------------------
# Self-test (no LLM / build / test wiring needed)
# ---------------------------------------------------------------------------

_SELFTEST_NAIVE_OCL = r"""
__kernel void matmul_naive(__global float* A, __global float* B, __global float* C,
                            int M, int N, int K) {
    int gx = get_global_id(0);
    int gy = get_global_id(1);
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) acc += A[gy*K + k] * B[k*N + gx];
    C[gy*N + gx] = acc;
}
"""

_SELFTEST_TILED_OCL = r"""
__kernel void matmul_tiled(__global float* restrict A, __global float* restrict B,
                            __global float* restrict C, int M, int N, int K) {
    __local float tileA[16][17];
    __local float tileB[16][17];
    int lx = get_local_id(0), ly = get_local_id(1);
    int gx = get_group_id(0)*16 + lx;
    int gy = get_group_id(1)*16 + ly;
    float acc = 0.0f;
    for (int kt = 0; kt < K; kt += 16) {
        tileA[ly][lx] = A[gy*K + (kt + lx)];
        tileB[ly][lx] = B[(kt + ly)*N + gx];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int k = 0; k < 16; ++k) acc += tileA[ly][k] * tileB[k][lx];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    C[gy*N + gx] = acc;
}
"""


def _self_test() -> int:
    print("[1] classify_code on naive OCL matmul (expect all zeros except maybe parallelism=0):")
    naive = classify_code(_SELFTEST_NAIVE_OCL)
    print("    ->", naive)
    assert naive[0] <= 1, naive

    print("[2] classify_code on tiled OCL matmul (expect memory>=2, parallelism>=1):")
    tiled = classify_code(_SELFTEST_TILED_OCL)
    print("    ->", tiled)
    assert tiled[0] >= 2, tiled
    assert tiled[2] >= 1, tiled

    print("[3] Archive add/elite competition:")
    arch = Archive()
    p1 = Program(id="p1", code=_SELFTEST_NAIVE_OCL, score=5.0)
    p2 = Program(id="p2", code=_SELFTEST_TILED_OCL, score=20.0)
    e1 = arch.add(p1)
    e2 = arch.add(p2)
    print(f"    p1 elite={e1}, p2 elite={e2}, cells filled={len(arch.feature_map)}")
    assert e1 and e2

    print("[4] pick_target_profile (mutate from naive):")
    underx = arch.underexplored_cells()
    target = pick_target_profile(p1, "mutate", underx)
    print("    ->", target)

    print("[5] render_optimization_instructions for (mem=2, par=2, ocl):")
    txt = render_optimization_instructions((2, 0, 2, 0), "ocl")
    assert "Local Memory Tiling" in txt and "Sub-Group Collectives" in txt, txt
    print("    -> OK ({} chars)".format(len(txt)))

    print("[6] build_prompt round-trip (no LLM needed):")
    prompt = build_prompt(
        reference_code=_SELFTEST_NAIVE_OCL,
        parent=Program(id="parent", code=_SELFTEST_TILED_OCL, score=20.0,
                       runtime_ms=2.0, correct=True, compiled=True,
                       eval_log="all good"),
        inspirations=[],
        target=(2, 0, 2, 0), backend="ocl",
        hardware="Intel B580 / 20 Xe2 cores / 64KB SLM",
        recent_compile_errors=["error: 'foo' undeclared"],
    )
    assert "Required Optimizations" in prompt
    assert "Avoid these compile errors" in prompt
    assert "Previous Version" in prompt
    print("    -> OK ({} chars)".format(len(prompt)))

    print("[7] TransitionTracker hints from synthetic history:")
    tracker = TransitionTracker()
    parent = Program(id="par", code="", coords=(1, 1, 1, 0), score=5.0)
    for _ in range(5):
        child = Program(id="ch", code="", coords=(2, 1, 1, 0), score=10.0)
        tracker.record(parent, child, became_elite=True)
    hints = tracker.mutation_hints_for(parent)
    print("    ->", hints)
    assert hints, "expected at least one mutation hint"
    assert "memory_opt" in hints[0]

    print("[8] combined_score sanity:")
    er = EvalResult(perf_score=5, correct=True, runtime_ms=1.0, speedup=27.9)
    s = combined_score(er)
    print(f"    correct + 27.9x speedup -> {s:.2f} (expected ≈ 88.7)")
    assert 88.0 < s < 89.0

    print("\nAll self-tests passed.")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--self-test", action="store_true",
                        help="Run the algorithm-level self-test (no LLM/build needed).")
    parser.add_argument("--run", action="store_true",
                        help="Run the full loop (requires LLM/build/test stubs to be implemented).")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(message)s",
    )

    if args.self_test:
        return _self_test()
    if args.run:
        # Wire your `build`, `run_correctness_test`, `run_perf_test`,
        # `llm_complete` first; then call `run(reference_code=...,
        # reference_runtime_ms=...)`.
        print("Wire the four stubs at the top of this file (build, "
              "run_correctness_test, run_perf_test, llm_complete) and "
              "call `run(reference_code, reference_runtime_ms, ...)`.")
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(_main())
