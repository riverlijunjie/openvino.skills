"""
optimization_classifier.py -- deterministic code -> MAP-Elites cell.

Faithful, dependency-light distillation of
kernelfoundry/algorithm/evolve_database_optimization_aware.py::OptimizationFeatureClassifier
and utils/map_elites_patterns.py.

A kernel's *behavior descriptor* is (memory_opt, compute_opt, parallelism_opt[, esimd_opt]),
each an integer level 0..3 found by regex pattern matching on the source. The mapping is
deterministic and uses only the code text -- runtime metrics never change a kernel's cell, they
only decide which kernel wins a cell (see references/algorithm.md sections 3-5).

Scoring per dimension (matches the original):
  - for each level L in {1,2,3}: score = matched_pattern_weight / total_weight_at_L
    (with a floor of 0.25 if any high-weight >=0.9 pattern matched)
  - walk levels top-down (3->1); return the first L whose score >= 0.15 + (L-1)*0.03; else 0.

Usage:
    from optimization_classifier import classify
    coords = classify(source_code, backend="cuda", include_esimd=False)  # -> (m, c, p)
"""
from __future__ import annotations
import re
from typing import Dict, List, Tuple

# ----------------------------------------------------------------------------------------------
# Pattern ladders. {dimension: {backend: {level: [(weight, regex), ...]}}}. Level 0 is implicit.
# These mirror the *signals* the original classifier keys on; extend per backend as needed.
# ----------------------------------------------------------------------------------------------
PATTERNS: Dict[str, Dict[str, Dict[int, List[Tuple[float, str]]]]] = {
    "memory": {
        "cuda": {
            1: [(1.0, r"\bfloat4\b|\bfloat2\b|\bdouble2\b"), (0.9, r"reinterpret_cast<\s*float4"),
                (0.6, r"__ldg\s*\(")],
            2: [(1.0, r"__shared__\b"), (0.8, r"__syncthreads\s*\("),
                (0.5, r"\[\s*\w+\s*\+\s*1\s*\]")],  # +1 padding
            3: [(1.0, r"#pragma\s+unroll"), (0.9, r"\bregister\b|acc\[\s*\w+\s*\]\[\s*\w+\s*\]"),
                (0.8, r"double[_\s]?buffer|cp\.async|__pipeline")],
        },
        "sycl": {
            1: [(1.0, r"sycl::vec<|::vec<\s*float"), (0.8, r"\.load\s*\(|\.store\s*\(")],
            2: [(1.0, r"local_accessor<"), (0.8, r"group_barrier\s*\("),
                (0.6, r"group_local_memory")],
            3: [(1.0, r"async_work_group_copy|joint_prefetch"), (0.8, r"#pragma\s+unroll")],
        },
        "triton": {
            1: [(1.0, r"tl\.load\s*\(.*mask"), (0.6, r"tl\.arange")],
            2: [(1.0, r"tl\.make_block_ptr")],
            3: [(1.0, r"tl\.advance"), (0.8, r"num_stages\s*=\s*[3-9]")],
        },
        "opencl": {
            1: [(1.0, r"\bfloat4\b|\bfloat8\b"), (0.8, r"vload\d|vstore\d")],
            2: [(1.0, r"__local\b"), (0.8, r"barrier\s*\(\s*CLK_LOCAL")],
            3: [(1.0, r"async_work_group_copy"), (0.8, r"#pragma\s+unroll")],
        },
    },
    "compute": {
        "cuda": {
            1: [(1.0, r"\bfmaf?\s*\(|__fmaf"), (0.5, r"// *fus|fused")],
            2: [(1.0, r"running_max|m_i\b|online|welford|kahan"), (0.8, r"exp\s*\(\s*\w+\s*-\s*\w*max")],
            3: [(1.0, r"flash|block.*attention|joint_matrix|wmma"), (0.7, r"acc\b.*rescale|l_i\b")],
        },
        "sycl": {
            1: [(1.0, r"sycl::fma\s*\(")],
            2: [(1.0, r"reduce_over_group|online|welford"), (0.8, r"running_max|m_i\b")],
            3: [(1.0, r"joint_matrix|flash|block.*attention")],
        },
        "triton": {
            1: [(0.8, r"tl\.math|libdevice")],
            2: [(1.0, r"tl\.max\s*\(|tl\.sum\s*\("), (0.8, r"m_i\b|online")],
            3: [(1.0, r"tl\.dot\s*\(.*acc|flash"), (0.8, r"l_i\b|acc\b")],
        },
        "opencl": {
            1: [(1.0, r"\bmad\s*\(|\bfma\s*\(|native_")],
            2: [(1.0, r"sub_group_reduce|online|welford")],
            3: [(1.0, r"flash|block.*accum")],
        },
    },
    "parallelism": {
        "cuda": {
            1: [(1.0, r"__shared__.*reduc|tree.*reduc"), (0.6, r"for\s*\(.*>>=\s*1|/=\s*2")],
            2: [(1.0, r"__shfl_(xor|down|up)_sync|__shfl_sync")],
            3: [(1.0, r"warpId|laneId|warp_tile|block.*warp.*thread")],
        },
        "sycl": {
            1: [(1.0, r"group_barrier.*reduc|tree")],
            2: [(1.0, r"reduce_over_group|group_broadcast|shift_group_(left|right)")],
            3: [(1.0, r"parallel_for_work_group|sub_group.*tile")],
        },
        "triton": {
            1: [(0.8, r"tl\.sum\s*\(|tl\.max\s*\(")],
            2: [(0.8, r"num_warps\s*=")],
            3: [(1.0, r"@triton\.autotune|tl\.num_programs")],
        },
        "opencl": {
            1: [(1.0, r"barrier.*reduc|tree")],
            2: [(1.0, r"sub_group_reduce|sub_group_broadcast|intel_sub_group_shuffle")],
            3: [(1.0, r"reqd_work_group_size|work_group.*sub_group")],
        },
    },
    "esimd": {
        "cuda": {
            1: [(1.0, r"wmma::|nvcuda::wmma")],
            2: [(1.0, r"load_matrix_sync|mma_sync")],
            3: [(1.0, r"cp\.async.*mma|pipelined.*mma")],
        },
        "sycl": {
            1: [(1.0, r"simd<|sycl_explicit_simd")],
            2: [(1.0, r"block_load|block_store|lsc_")],
            3: [(1.0, r"\bdpas\b|raw_send")],
        },
        "triton": {1: [(0.0, r"$^")], 2: [(0.0, r"$^")], 3: [(0.0, r"$^")]},
        "opencl": {
            1: [(1.0, r"intel_sub_group_block_read")],
            2: [(1.0, r"intel_sub_group_block_write")],
            3: [(1.0, r"cl_intel_subgroup_matrix|dpas")],
        },
    },
}

_THRESH = lambda level: 0.15 + (level - 1) * 0.03  # noqa: E731


def _level_score(code: str, level_patterns: List[Tuple[float, str]]) -> float:
    if not level_patterns:
        return 0.0
    total = sum(w for w, _ in level_patterns)
    matched = 0.0
    high_hit = False
    for w, rx in level_patterns:
        if re.search(rx, code, re.IGNORECASE):
            matched += w
            if w >= 0.9:
                high_hit = True
    base = matched / total if total else 0.0
    if high_hit:
        base = max(base, 0.25)
    return base


def _classify_dimension(code: str, ladder: Dict[int, List[Tuple[float, str]]]) -> int:
    for level in (3, 2, 1):
        if _level_score(code, ladder.get(level, [])) >= _THRESH(level):
            return level
    return 0


def classify(code: str, backend: str = "cuda", include_esimd: bool = False) -> Tuple[int, ...]:
    """Return the behavior-descriptor coordinates for `code` on the given backend."""
    backend = backend.lower()
    dims = ["memory", "compute", "parallelism"] + (["esimd"] if include_esimd else [])
    coords = []
    for dim in dims:
        ladder = PATTERNS[dim].get(backend, {})
        coords.append(_classify_dimension(code, ladder))
    return tuple(coords)


if __name__ == "__main__":
    import sys, json
    src = open(sys.argv[1]).read() if len(sys.argv) > 1 else sys.stdin.read()
    backend = sys.argv[2] if len(sys.argv) > 2 else "cuda"
    c = classify(src, backend=backend, include_esimd=False)
    print(json.dumps({"backend": backend, "coords": c,
                      "dims": ["memory_opt", "compute_opt", "parallelism_opt"]}))
