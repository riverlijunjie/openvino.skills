"""
evolve_loop.py -- a faithful, dependency-light reimplementation of the KernelFoundry
evolutionary kernel-optimization algorithm.

It distills, in one file, the pieces documented in references/algorithm.md:
  * MAP-Elites archive over (memory_opt, compute_opt, parallelism_opt[, esimd_opt]) cells
  * island model with ring migration
  * correctness-gated speedup fitness (perf_score 0..5 + speedup bonus)
  * parent sampling: exploitation (elite softmax) / exploration (island) / random
  * diverse inspirations with an optimization-level bonus
  * target-cell selection (under-explored region OR mutate-the-parent strategies)
  * optimization-aware prompt assembly that demands the target cell's techniques
  * optional lightweight QD-gradient steering
  * OPT-IN advanced convergence (references/advanced_convergence.md), all default OFF:
      ① fine-grained / measured / CVT placement descriptors
      ② staged coarse→fine descriptor activation (with archive re-projection)
      ③ surrogate pre-ranking (generate k×B, evaluate top-B)
      ④a parameter trust region (fix structure, tune template params)
      ④b edit trust region (SEARCH/REPLACE diffs; reject out-of-cell mutations pre-eval)

You supply callbacks:
    generate(prompt:str) -> str          # your LLM; return its raw answer (code or fenced code)
    evaluate(code:str)  -> EvalResult     # compile + test + benchmark; return the result
    eval_params(center, params) -> EvalResult   # (④a only) eval a structure at given template params
    edit_fn(prompt:str) -> str            # (④b only) your LLM in diff mode; returns SEARCH/REPLACE

Run the built-in demos (no LLM/GPU needed):
    python evolve_loop.py --demo            # baseline loop
    python evolve_loop.py --demo-advanced   # ①②③④a together
    python evolve_loop.py --demo-edit-tr    # ④b edit trust region (stay-in-cell)

Wire it to a real backend by importing run_evolution() and passing your callbacks.
This mirrors the loop in kernelfoundry/algorithm/controller.py::_run_single.
"""
from __future__ import annotations

import argparse
import heapq
import json
import math
import os
import random
import re
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from optimization_classifier import classify

_KNOWLEDGE = json.loads((Path(__file__).parent / "optimization_knowledge.json").read_text())
DIMS = ["memory_opt", "compute_opt", "parallelism_opt"]
_KNOWLEDGE_KEY = {"memory_opt": "memory", "compute_opt": "compute", "parallelism_opt": "parallelism"}


# ====================================================================================
# Genome + fitness  (schemas.py::EvalResult / Program)
# ====================================================================================
@dataclass
class EvalResult:
    compiled: bool = False
    correct: bool = False
    perf_score: int = -1            # 0 syntax,1 not-compiled,2 runtime,3 shape,4 value,5 correct
    runtime: float = -1.0           # ms
    runtime_improvement: float = -1.0  # gmean(ref/custom); -1 means "no reference"
    log: str = ""
    # Optional profiler-measured features for MEASURED descriptors (strategy ①). Populate from
    # profiler_feedback (occupancy, gmem/slm bandwidth %, ve/sm stall %, arithmetic intensity, ...).
    # Each value should be normalized to ~[0,1]. Empty → measured descriptors fall back to static.
    features: dict = field(default_factory=dict)

    def score(self) -> float:
        """Single source of truth (EvalResult.compute_performance_score)."""
        assert self.perf_score != -1, "must evaluate before scoring"
        s = float(self.perf_score)
        if self.runtime_improvement > 0:
            s += self.runtime_improvement
        elif self.runtime_improvement == -1 and self.correct and self.runtime > 0:
            s += 1.0 / self.runtime
        return s


@dataclass
class Program:
    id: str
    code: str
    parent_id: Optional[str] = None
    generation: int = 0
    iteration_found: int = 0
    coords: Tuple[int, ...] = ()        # STATIC ladder descriptor: targeting, gradient, inspirations
    cell: Tuple[int, ...] = ()          # PLACEMENT key in the archive (== coords for static descriptor)
    result: Optional[EvalResult] = None
    island: int = 0
    via_surrogate: bool = False         # strategy ③: was this child pre-ranked by the surrogate?

    @property
    def fitness(self) -> float:
        return self.result.score() if self.result else -math.inf


# ====================================================================================
# Descriptor spaces  (strategy ①: fine-grained / measured / CVT-bounded)
# ------------------------------------------------------------------------------------
# A Descriptor turns a (code, EvalResult) pair into an archive cell key. Three variants:
#   * StaticDescriptor   -- the original code-pattern ladder (a-priori, pre-evaluation).
#   * MeasuredDescriptor -- bins profiler-measured features that correlate with runtime
#                           (a-posteriori); falls back to static when features are absent.
#   * CVTDescriptor      -- wraps any continuous featurizer; k fixed Voronoi centroids cap the
#                           archive size so dimensionality can grow without cell explosion.
# Placement uses these AFTER evaluation; target selection still uses the STATIC ladder before
# generation (see references/advanced_convergence.md "two-descriptor split").
# ====================================================================================
class StaticDescriptor:
    """Original behavior descriptor: code-pattern levels on each axis (4 levels)."""

    def __init__(self, backend: str, include_esimd: bool = False):
        self.backend, self.include_esimd, self.levels, self.ndim = backend, include_esimd, 4, 3

    def of(self, code: str, result: "EvalResult") -> Tuple[int, ...]:
        return classify(code, self.backend, self.include_esimd)


class MeasuredDescriptor:
    """Bin profiler-measured features (occupancy × arithmetic-intensity × DRAM-BW%, ...).

    `feature_keys` name entries in EvalResult.features (normalized ~[0,1]); each is split into
    `bins` buckets. This is the recommended fine-grained descriptor: it is directly
    performance-correlated, so 'climbing' the archive tracks runtime instead of drifting from it.
    Falls back to the static descriptor whenever features are missing (e.g. kernel didn't run)."""

    def __init__(self, backend: str, feature_keys=("occupancy", "arithmetic_intensity", "dram_bw_pct"),
                 bins: int = 6):
        self.backend = backend
        self.feature_keys = tuple(feature_keys)
        self.bins = bins
        self.ndim = len(self.feature_keys)
        self._static = StaticDescriptor(backend)

    def of(self, code: str, result: "EvalResult") -> Tuple[int, ...]:
        feats = getattr(result, "features", None) or {}
        if not all(k in feats for k in self.feature_keys):
            return self._static.of(code, result)  # graceful fallback (uncompiled / not-run)
        out = []
        for k in self.feature_keys:
            v = min(0.999999, max(0.0, float(feats[k])))
            out.append(int(v * self.bins))
        return tuple(out)


class CVTDescriptor:
    """CVT-MAP-Elites: map a continuous featurizer onto k fixed Voronoi centroids.

    Archive size is bounded by `k` regardless of descriptor dimensionality, which is what makes
    *fine-grained* descriptors affordable. Centroids are sampled deterministically from the unit
    cube via the run's RNG (a faithful, lightweight stand-in for Lloyd-relaxed CVT centroids)."""

    def __init__(self, featurizer: Callable[[str, "EvalResult"], Tuple[float, ...]],
                 ndim: int, k: int, rng: random.Random):
        self.featurizer = featurizer
        self.ndim = ndim
        self.k = k
        self.centroids = [tuple(rng.random() for _ in range(ndim)) for _ in range(k)]

    def of(self, code: str, result: "EvalResult") -> Tuple[int, ...]:
        x = self.featurizer(code, result)
        best_i, best_d = 0, math.inf
        for i, c in enumerate(self.centroids):
            d = sum((a - b) ** 2 for a, b in zip(x, c))
            if d < best_d:
                best_i, best_d = i, d
        return (best_i,)  # cell key = centroid index


def make_descriptor(cfg: dict, backend: str, rng: random.Random):
    """Build the placement descriptor from cfg['descriptor'] (default: original static ladder)."""
    spec = cfg.get("descriptor", "static")
    if spec == "static":
        return StaticDescriptor(backend, include_esimd=cfg.get("enable_esimd_exploration", False))
    if spec == "measured":
        return MeasuredDescriptor(backend, feature_keys=cfg.get("measured_feature_keys",
                                  ("occupancy", "arithmetic_intensity", "dram_bw_pct")),
                                  bins=cfg.get("measured_bins", 6))
    if spec == "cvt":
        md = MeasuredDescriptor(backend, feature_keys=cfg.get("measured_feature_keys",
                                ("occupancy", "arithmetic_intensity", "dram_bw_pct")),
                                bins=cfg.get("measured_bins", 6))

        def featurizer(code, result):
            feats = getattr(result, "features", None) or {}
            if all(k in feats for k in md.feature_keys):
                return tuple(min(1.0, max(0.0, float(feats[k]))) for k in md.feature_keys)
            coords = classify(code, backend)            # static fallback, normalized to [0,1]
            return tuple(c / 3.0 for c in coords) + (0.0,) * (len(md.feature_keys) - len(coords))
        return CVTDescriptor(featurizer, ndim=len(md.feature_keys),
                             k=cfg.get("cvt_cells", 512), rng=rng)
    raise ValueError(f"unknown descriptor spec: {spec!r}")


# ====================================================================================
# QD gradient (qd_gradient.py) -- lightweight: per-direction success stats + empty-cell pull
# ====================================================================================
class QDGradient:
    def __init__(self, cfg: dict):
        self.fitness_w = cfg.get("fitness_weight", 0.4)
        self.rate_w = cfg.get("improvement_rate_weight", 0.4)
        self.explore_w = cfg.get("exploration_weight", 0.2)
        self.history = deque(maxlen=cfg.get("max_history", 10000))
        # per dimension: list over axis-direction sign of [successes, total, sum_delta]
        self.dir_stats: Dict[int, Dict[int, List[float]]] = {d: {-1: [0, 0, 0.0], 1: [0, 0, 0.0]} for d in range(len(DIMS))}

    def record(self, pc: Tuple[int, ...], cc: Tuple[int, ...], df: float):
        self.history.append((pc, cc, df))
        for d in range(len(DIMS)):
            step = cc[d] - pc[d]
            if step == 0:
                continue
            sign = 1 if step > 0 else -1
            st = self.dir_stats[d][sign]
            st[1] += 1
            st[2] += df
            if df > 0:
                st[0] += 1

    def gradient(self, occupied: set, grid_levels: int) -> Tuple[float, ...]:
        g = []
        for d in range(len(DIMS)):
            pos, neg = self.dir_stats[d][1], self.dir_stats[d][-1]
            fit = (pos[2] / pos[1] if pos[1] else 0.0) - (neg[2] / neg[1] if neg[1] else 0.0)
            rate = (pos[0] / pos[1] if pos[1] else 0.0) - (neg[0] / neg[1] if neg[1] else 0.0)
            # exploration: do higher levels on this axis tend to be empty?
            filled = sum(1 for c in occupied if len(c) > d and c[d] >= grid_levels - 1)
            explore = 1.0 / (1.0 + filled)
            g.append(self.fitness_w * fit + self.rate_w * rate + self.explore_w * explore)
        return tuple(g)


# ====================================================================================
# MAP-Elites archive + island model  (OptimizationAwareDatabase)
# ====================================================================================
class Archive:
    def __init__(self, cfg: dict, rng: random.Random, descriptor=None):
        self.cfg = cfg
        self.rng = rng
        self.levels = 4
        self.descriptor = descriptor                          # placement descriptor (strategy ①)
        self.programs: Dict[str, Program] = {}
        self.feature_map: Dict[Tuple[int, ...], str] = {}     # cell -> elite id
        self.num_islands = cfg.get("num_islands", 4)
        self.islands: List[set] = [set() for _ in range(self.num_islands)]
        self.active_island = 0
        self._since_switch = 0
        self.generation = 0
        self.gradient = QDGradient(cfg.get("gradient_config", {})) if cfg.get("use_gradient_tracking", True) else None

    def _place(self, prog: Program) -> Tuple[int, ...]:
        """Compute the archive cell key. Uses the configured descriptor; defaults to static coords."""
        if self.descriptor is not None:
            return self.descriptor.of(prog.code, prog.result)
        return prog.coords

    # ---- insert -------------------------------------------------------------------
    def add(self, prog: Program, iteration: int) -> bool:
        prog.iteration_found = iteration
        prog.island = self.active_island
        prog.cell = self._place(prog)
        self.programs[prog.id] = prog
        self.islands[self.active_island].add(prog.id)
        became_elite = False
        cur = self.feature_map.get(prog.cell)
        if cur is None or prog.fitness > self.programs[cur].fitness:
            self.feature_map[prog.cell] = prog.id
            became_elite = True
        # gradient is steered by the STATIC ladder (a-priori, what targeting can act on)
        if self.gradient is not None and prog.parent_id in self.programs:
            par = self.programs[prog.parent_id]
            self.gradient.record(par.coords, prog.coords, prog.fitness - par.fitness)
        self._enforce_population_limit()
        # island bookkeeping
        self._since_switch += 1
        if self._since_switch >= self.cfg.get("programs_per_island", 10):
            self.active_island = (self.active_island + 1) % self.num_islands
            self._since_switch = 0
            self.generation += 1
            if self.generation % self.cfg.get("migration_interval", 10) == 0:
                self._migrate()
        return became_elite

    def _enforce_population_limit(self):
        cap = self.cfg.get("population_size", 1000)
        if len(self.programs) <= cap:
            return
        elites = set(self.feature_map.values())
        best = self.best().id if self.feature_map else None
        removable = [(p.fitness, pid) for pid, p in self.programs.items() if pid not in elites and pid != best]
        for _, pid in heapq.nsmallest(len(self.programs) - cap, removable):
            self.programs.pop(pid, None)
            for isl in self.islands:
                isl.discard(pid)

    def _migrate(self):
        rate = self.cfg.get("migration_rate", 0.1)
        for i, isl in enumerate(self.islands):
            members = sorted(isl, key=lambda pid: self.programs[pid].fitness, reverse=True)
            n = max(1, int(len(members) * rate))
            self.islands[(i + 1) % self.num_islands].update(members[:n])  # copy-mode

    def reproject(self, descriptor):
        """Swap the placement descriptor and rebuild feature_map from all stored programs.

        Needed by staged dimension activation (strategy ②): when coarse→fine descriptors turn on,
        every archived elite is re-classified under the finer space (code+result are stored, so
        this is cheap — no re-evaluation). The best program per new cell is kept."""
        self.descriptor = descriptor
        self.feature_map = {}
        for prog in sorted(self.programs.values(), key=lambda p: p.fitness, reverse=True):
            prog.cell = self._place(prog)
            cur = self.feature_map.get(prog.cell)
            if cur is None or prog.fitness > self.programs[cur].fitness:
                self.feature_map[prog.cell] = prog.id

    # ---- query --------------------------------------------------------------------
    def best(self) -> Program:
        return max(self.programs.values(), key=lambda p: p.fitness)

    def occupied(self) -> set:
        return set(self.feature_map.keys())

    def elites(self) -> List[Program]:
        return [self.programs[i] for i in self.feature_map.values()]

    # ---- sampling (sample / _sample_parent / _sample_diverse_inspirations) --------
    def sample(self) -> Tuple[Program, List[Program]]:
        parent = self._sample_parent()
        insp = self._sample_inspirations(parent)
        return parent, insp

    def _sample_parent(self) -> Program:
        r = self.rng.random()
        expl = self.cfg.get("exploration_ratio", 0.2)
        expt = self.cfg.get("exploitation_ratio", 0.7)
        if r < expt and self.feature_map:                       # exploitation: elite softmax
            elites = self.elites()
            scores = [e.fitness for e in elites]
            m = max(scores)
            w = [math.exp(s - m) for s in scores]
            return self.rng.choices(elites, weights=w, k=1)[0]
        if r < expt + expl and self.islands[self.active_island]:  # exploration: current island
            pool = [self.programs[i] for i in self.islands[self.active_island]]
            return self.rng.choice(pool)
        return self.rng.choice(list(self.programs.values()))     # random

    def _sample_inspirations(self, parent: Program) -> List[Program]:
        n = self.cfg.get("num_inspirations", 2)
        parent_sum = sum(parent.coords)
        pool = [e for e in self.elites() if e.coords != parent.coords and e.id != parent.id]
        # optimization-level bonus: prefer elites that are *more* optimized than the parent
        pool.sort(key=lambda e: e.fitness + max(0, sum(e.coords) - parent_sum) * 0.1, reverse=True)
        insp = pool[:n]
        if len(insp) < n:  # fall back to global top programs
            rest = sorted((p for p in self.programs.values() if p.id != parent.id and p not in insp),
                          key=lambda p: p.fitness, reverse=True)
            insp += rest[: n - len(insp)]
        return insp


# ====================================================================================
# Target-cell selection (_get_target_optimization_profile) + prompt assembly
# ====================================================================================
def pick_target_cell(parent: Program, archive: Archive, cfg: dict, rng: random.Random) -> Tuple[int, ...]:
    levels, ndim = 4, len(DIMS)
    guidance_explore = cfg.get("guidance_exploration_rate", 0.0)
    parent_correct = parent.result and parent.result.correct
    # bias from the QD gradient (which axis is paying off / under-explored)
    grad = archive.gradient.gradient(archive.occupied(), levels) if archive.gradient else (0.0,) * ndim
    blend = cfg.get("gradient_sampling_weight", 0.3)

    if parent_correct and rng.random() > guidance_explore:
        # aim at an under-explored region, nudged by the gradient
        empty = [c for c in _all_cells(levels, ndim) if c not in archive.occupied()]
        if empty and rng.random() < blend + 0.5:
            empty.sort(key=lambda c: sum(grad[d] * c[d] for d in range(ndim)), reverse=True)
            return empty[0]
    # otherwise mutate the parent's profile per strategy
    strat = cfg.get("exploration_strategy", "mutate")
    coords = list(parent.coords) if parent.coords else [0] * ndim
    if strat == "diversify":
        for d in range(ndim):
            coords[d] = min(levels - 1, max(0, coords[d] + rng.choice([-1, 0, 1])))
    elif strat == "intensify":
        d = min(range(ndim), key=lambda i: coords[i])           # push the weakest axis up
        coords[d] = min(levels - 1, coords[d] + 1)
    else:  # "mutate": +1 on a single (gradient-preferred) axis
        d = max(range(ndim), key=lambda i: grad[i]) if archive.gradient else rng.randrange(ndim)
        coords[d] = min(levels - 1, coords[d] + 1)
    return tuple(coords)


def _all_cells(levels: int, ndim: int) -> List[Tuple[int, ...]]:
    cells = [()]
    for _ in range(ndim):
        cells = [c + (l,) for c in cells for l in range(levels)]
    return cells


def build_prompt(task: dict, parent: Program, inspirations: List[Program],
                 target: Tuple[int, ...], backend: str) -> str:
    """Assemble the optimization-aware prompt (main_prompt.j2 + build_exploration_prompt)."""
    lang = task["language"]
    status = "translate" if (parent.code.strip() == "" or parent.generation == 0 and not parent.result) \
        else ("error" if parent.result and not parent.result.correct else "correct")
    out = [f"# You are a {lang} GPU kernel optimization expert.",
           f"Given the reference below, produce a performant, functionally-identical {lang} kernel.",
           "", "## Reference implementation:", "```", task["reference"], "```", ""]
    if task.get("user_instructions"):
        out += ["## User instructions:", task["user_instructions"], ""]
    for i, ins in enumerate(inspirations, 1):
        r = ins.result
        tag = f"score {r.score():.2f}, {'correct' if r.correct else 'incorrect'}" if r else "untested"
        out += [f"### Prior version {i} ({tag}):", f"```{lang}", ins.code, "```", ""]
    if parent.code.strip():
        out += ["### Last kernel:", f"```{lang}", parent.code, "```"]
        if parent.result and parent.result.log:
            out += ["Evaluation feedback:", parent.result.log, ""]
    # --- the optimization-aware "Required Optimizations" section, from the target cell ---
    out += ["## Required Optimizations (apply these specific techniques):"]
    for d, level in zip(DIMS, target):
        if level <= 0:
            continue
        text = _KNOWLEDGE.get(_KNOWLEDGE_KEY[d], {}).get(backend, {}).get(str(level), "")
        if text:
            out += [f"- **{d} -> level {level}**: {text}"]
    out += ["", "## Critical requirements:",
            "1. Match the reference's functionality exactly.",
            "2. The code must compile and run on the target GPU.",
            f"3. Provide the complete {lang} code in a single ```{lang} code block."]
    return "\n".join(out)


def extract_code(answer: str, lang: str) -> str:
    """extract_code_flexible: fenced block -> tagged -> raw."""
    m = re.search(rf"```(?:{lang}|cuda|cpp|c\+\+|sycl|opencl|ocl|python|triton)?\s*\n(.*?)```",
                  answer, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(rf"<{lang}>(.*?)</{lang}>", answer, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return answer.strip()


# ====================================================================================
# Surrogate pre-ranking  (strategy ③: SAIL / DSA-ME -- generate k×B, evaluate top-B)
# ------------------------------------------------------------------------------------
# The expensive resource is `evaluate` (compile+test+benchmark+profile). A surrogate scores cheap
# candidate CODE so only the most promising ones are evaluated. Default surrogate is a transparent
# heuristic (compile-likelihood × climbed-levels) so the demo needs no ML; swap in a learned ranker
# or an LLM-judge via cfg['surrogate']. Honesty knobs below mirror the doc's pitfalls.
# ====================================================================================
class HeuristicSurrogate:
    """Cheap, dependency-free stand-in: prefers code that is likely to compile and that climbs the
    optimization ladder, lightly penalizing known antipattern strings. Returns a score; higher first."""

    def __init__(self, backend: str):
        self.backend = backend
        anti = _KNOWLEDGE.get("antipatterns", {}).get(backend, "")
        # a few coarse antipattern signals pulled from the knowledge base text
        self._anti = [s.strip().lower()[:18] for s in re.split(r"[;.]", anti) if len(s.strip()) > 8][:6]

    def score(self, code: str) -> float:
        if not code or "= ;" in code or code.count("{") != code.count("}"):
            return -1.0                                   # almost surely won't compile
        coords = classify(code, self.backend)
        s = float(sum(coords))                            # reward climbing the ladder
        low = code.lower()
        s -= 0.5 * sum(1 for a in self._anti if a and a in low)
        return s


def _rank_candidates(codes: List[str], surrogate, k_keep: int, rng: random.Random) -> List[int]:
    """Return indices of the top-k_keep candidates by surrogate score (acquisition = top-k, not
    top-1, to hedge surrogate bias). Ties broken randomly for diversity."""
    scored = [(surrogate.score(c), rng.random(), i) for i, c in enumerate(codes)]
    scored.sort(reverse=True)
    return [i for _, _, i in scored[:k_keep]]


# ====================================================================================
# Parameter trust-region local refinement  (strategy ④a -- the convergence workhorse)
# ------------------------------------------------------------------------------------
# MAP-Elites illuminates STRUCTURE; once a strong elite is found, we FIX the structure and tune its
# continuous template parameters (TILE_M/N, BLOCK, num_stages, num_warps, unroll) inside a shrinking
# trust region (TuRBO-style pattern search). This is what actually converges to the local optimum,
# which plain MAP-Elites is not designed to do. Requires the task to expose a parametric evaluator.
# ====================================================================================
def trust_region_refine(center: Program,
                        param_space: Dict[str, Tuple[float, float]],
                        eval_params: Callable[[Program, Dict[str, float]], EvalResult],
                        cfg: dict, rng: random.Random, new_id: Callable[[], str], logfn) -> Program:
    """Pattern-search trust region over `param_space` (name -> (lo, hi)), starting from center.

    eval_params(center, params) compiles+benchmarks the center's STRUCTURE with those params and
    returns an EvalResult. Returns the best program found (>= center). Budget: cfg['tr_max_evals'].
    `new_id` is a zero-arg id factory (e.g. _ThreadSafeCounter().next)."""
    names = list(param_space)
    lo = {n: param_space[n][0] for n in names}
    hi = {n: param_space[n][1] for n in names}
    # start at the center's known params if present (set by a prior refine), else mid-box
    prior = getattr(center, "_best_params", {})
    x = {n: float(prior.get(n, (lo[n] + hi[n]) / 2)) for n in names}
    radius = {n: cfg.get("tr_init_radius", 0.4) * (hi[n] - lo[n]) for n in names}
    best, best_res = dict(x), center.result
    best_score = best_res.score() if best_res else -math.inf
    budget = cfg.get("tr_max_evals", 16)
    shrink, grow = cfg.get("tr_shrink", 0.5), cfg.get("tr_grow", 1.6)
    min_r = {n: cfg.get("tr_min_radius", 0.02) * (hi[n] - lo[n]) for n in names}
    evals = 0
    while evals < budget and any(radius[n] > min_r[n] for n in names):
        improved = False
        # probe ± along each axis (pattern search poll)
        for n in names:
            for sign in (1, -1):
                if evals >= budget:
                    break
                cand = dict(best)
                cand[n] = min(hi[n], max(lo[n], best[n] + sign * radius[n]))
                if cand[n] == best[n]:
                    continue
                child = Program(id=new_id(), code=center.code, parent_id=center.id,
                                generation=center.generation + 1, coords=center.coords)
                child.result = eval_params(center, cand)
                evals += 1
                sc = child.result.score()
                logfn(f"    [tr] {n}{'+' if sign>0 else '-'}={cand[n]:.3g} score={sc:.3f}")
                if sc > best_score:
                    best, best_score, best_res = cand, sc, child.result
                    improved = True
        for n in names:
            radius[n] = radius[n] * (grow if improved else shrink)
    out = Program(id=new_id(), code=center.code, parent_id=center.id,
                  generation=center.generation + 1, coords=center.coords)
    out.result = best_res
    out._best_params = best  # noqa: keep for inspection
    return out


# ====================================================================================
# Edit trust-region local refinement  (strategy ④b -- derivative-free TR over the LLM operator)
# ------------------------------------------------------------------------------------
# A trust region whose VARIABLE is the LLM's diff edit and whose STEP SIZE is the edit *magnitude*.
# We reuse the SEARCH/REPLACE diff machinery (faithful to algorithm.utils.extract_code.extract_diffs
# / apply_diff): the model is asked for a bounded patch (<= `radius` changed lines); on improvement
# we GROW the budget, on failure we SHRINK it toward micro-edits. The defining constraint:
# every candidate is REJECTED BEFORE EVALUATION if its static descriptor leaves the center's cell,
# so the search stays in-cell (local) instead of wandering to a different structure. Pairs with ④a:
# ④b refines STRUCTURE locally, ④a then tunes that structure's continuous params.
# ====================================================================================
_DIFF_RE = re.compile(
    r"<{5,}\s*SEARCH\s*\n(?P<search>.*?)\n={5,}\s*\n(?P<replace>.*?)\n>{5,}\s*REPLACE",
    re.DOTALL,
)


def extract_diffs(diff_text: str) -> List[Tuple[str, str]]:
    """Parse SEARCH/REPLACE blocks → [(search, replace), ...] (faithful to extract_code.extract_diffs).

    Tolerant of fenced wrapping and >=5 markers, e.g.:
        <<<<<<< SEARCH
        old lines
        =======
        new lines
        >>>>>>> REPLACE
    """
    return [(m.group("search"), m.group("replace")) for m in _DIFF_RE.finditer(diff_text)]


def apply_diff(code: str, diff_text: str) -> Optional[str]:
    """Apply SEARCH/REPLACE blocks to `code` (first-occurrence replace, like apply_diff).

    Returns the patched code, or None if there are no blocks or any SEARCH text is not found
    (a malformed/non-applying patch is a failed mutation — handled as a shrink by the caller)."""
    blocks = extract_diffs(diff_text)
    if not blocks:
        return None
    out = code
    for search, replace in blocks:
        if search == "":                       # empty SEARCH = pure insertion at top (rare; allow)
            out = replace + "\n" + out
            continue
        if search not in out:
            return None                        # patch doesn't apply → caller shrinks the radius
        out = out.replace(search, replace, 1)
    return out


def edit_magnitude(diff_text: str) -> int:
    """Step-size proxy: number of changed lines across all blocks (search + replace line counts)."""
    n = 0
    for search, replace in extract_diffs(diff_text):
        n += search.count("\n") + 1 + replace.count("\n") + 1
    return n


def build_edit_prompt(task: dict, center: Program, radius_lines: int, backend: str) -> str:
    """Prompt the model for a BOUNDED diff that preserves the kernel's optimization structure.

    `radius_lines` is the trust-region step size: the max changed lines we permit this round. The
    instruction to keep the optimization profile constant is what biases the model toward in-cell
    edits (the hard rejection happens after, in the loop)."""
    lang = task["language"]
    prof = ", ".join(f"{d}={lvl}" for d, lvl in zip(DIMS, center.coords))
    fb = (center.result.log if center.result and center.result.log else "")
    return "\n".join([
        f"# You are a {lang} GPU kernel optimization expert performing a SMALL, LOCAL refinement.",
        f"The kernel below sits at optimization profile [{prof}]. Make it faster WITHOUT changing",
        "that profile: keep the same memory/compute/parallelism strategy (same tiling scheme, same",
        "synchronization structure, same level of vectorization). Tune only local details (indexing,",
        "unroll factors, fusion of adjacent ops, register reuse, bounds, constants).",
        "",
        f"## Current kernel:", f"```{lang}", center.code, "```",
        *( ["## Evaluation feedback:", fb, ""] if fb else [""] ),
        f"## Constraints:",
        f"- Respond ONLY with at most {radius_lines} changed lines, as SEARCH/REPLACE blocks:",
        "<<<<<<< SEARCH",
        "<exact lines to find>",
        "=======",
        "<replacement lines>",
        ">>>>>>> REPLACE",
        "- Do NOT introduce or remove a whole optimization technique (that would change the profile).",
        "- Keep the kernel correct and compilable.",
    ])


def edit_trust_region_refine(task: dict, center: Program,
                             edit_fn: Callable[[str], str],
                             evaluate: Callable[[str], EvalResult],
                             backend: str, cfg: dict, rng: random.Random,
                             new_id: Callable[[], str], logfn) -> Program:
    """Derivative-free trust region over LLM diff edits, constrained to stay in `center`'s cell.

    edit_fn(prompt) -> raw answer containing SEARCH/REPLACE blocks (your LLM, or a stub).
    Each round: ask for a patch of <= `radius` changed lines, apply it, REJECT (pre-eval) if the
    patched code's static descriptor != center's; else evaluate. Improvement → grow radius & move
    the center; failure/reject → shrink. Budget: cfg['etr_max_evals']."""
    radius = float(cfg.get("etr_init_radius", 8))          # trust region = max changed lines
    rmin, rmax = cfg.get("etr_min_radius", 2), cfg.get("etr_max_radius", 40)
    grow, shrink = cfg.get("etr_grow", 1.6), cfg.get("etr_shrink", 0.5)
    budget = cfg.get("etr_max_evals", 12)
    best = center
    best_score = center.fitness
    target_cell = center.coords                            # the cell we must stay in
    evals = rejected = 0
    while evals < budget and radius >= rmin:
        prompt = build_edit_prompt(task, best, int(round(radius)), backend)
        answer = edit_fn(prompt)
        patched = apply_diff(best.code, answer)
        mag = edit_magnitude(answer)
        if patched is None or patched == best.code:
            radius *= shrink                               # non-applying / empty patch → shrink
            logfn(f"    [etr] patch did not apply (mag={mag}); shrink radius→{radius:.1f}")
            continue
        # --- the constraint: reject out-of-cell mutations BEFORE paying for evaluation ---
        cand_cell = classify(patched, backend, include_esimd=cfg.get("enable_esimd_exploration", False))
        if cand_cell != target_cell:
            rejected += 1
            radius *= shrink
            logfn(f"    [etr] REJECT out-of-cell {cand_cell} != {target_cell} (mag={mag}); shrink→{radius:.1f}")
            continue
        if mag > radius * 1.5:                             # model ignored the budget → distrust, shrink
            radius *= shrink
            logfn(f"    [etr] over-budget edit (mag={mag} > {radius:.1f}); shrink→{radius:.1f}")
            continue
        child = Program(id=new_id(), code=patched, parent_id=best.id,
                        generation=best.generation + 1, coords=cand_cell)
        child.result = evaluate(patched)
        evals += 1
        sc = child.fitness
        ok = child.result.correct and child.result.compiled
        logfn(f"    [etr] in-cell edit mag={mag} score={sc:.3f} "
              f"{'correct' if ok else 'BAD'}; radius={radius:.1f}")
        if ok and sc > best_score:
            best, best_score = child, sc                   # move the center, grow the region
            radius = min(rmax, radius * grow)
        else:
            radius *= shrink                               # no improvement (or broke) → shrink
    logfn(f"    [etr] done: {evals} evals, {rejected} out-of-cell rejected, "
          f"best {center.fitness:.3f}→{best_score:.3f}")
    return best


# ====================================================================================
# The controller loop (controller.py::_run_single)
# ====================================================================================
def run_evolution(task: dict,
                  generate: Callable[[str], str],
                  evaluate: Callable[[str], EvalResult],
                  cfg: Optional[dict] = None,
                  seed_code: str = "",
                  eval_params: Optional[Callable[[Program, Dict[str, float]], EvalResult]] = None,
                  edit_fn: Optional[Callable[[str], str]] = None
                  ) -> Tuple[Program, Archive]:
    """Run the search. All advanced strategies are OPT-IN via cfg and default OFF; with them off the
    control flow is the original combined propose+evaluate loop (the demo's run-to-run variation is
    pre-existing: it shares one RNG across parallel branches). See references/advanced_convergence.md.

    cfg opt-ins:
      descriptor: 'static'|'measured'|'cvt'        (strategy ① fine-grained / bounded placement)
      stages: [{until_iter, descriptor, ...}, ...] (strategy ② coarse→fine activation)
      surrogate: True | <obj with .score(code)>    (strategy ③ pre-rank k×B, evaluate top-B)
      surrogate_overgen: 3                          (k multiplier for candidate over-generation)
      refine_every / refine_at_end + param_space   (strategy ④a trust-region param refinement)
      edit_refine_every / edit_refine_at_end       (strategy ④b edit trust region; needs edit_fn)
    """
    cfg = {**DEFAULT_CFG, **(task.get("cfg") or {}), **(cfg or {})}
    rng = random.Random(cfg.get("random_seed", 42))
    backend = task["language"].lower().replace("c++", "cpp")
    backend = {"ocl": "opencl"}.get(backend, backend)
    descriptor = make_descriptor(cfg, backend, rng)          # ① placement descriptor (default static)
    archive = Archive(cfg, rng, descriptor=descriptor)
    uid = _ThreadSafeCounter()                               # parallel-safe ids
    surrogate = _make_surrogate(cfg, backend)                # ③ None unless enabled
    stages = list(cfg.get("stages", []))                     # ② sorted, consumed as iters advance

    # iteration 0: seed (reference scaffold / provided best). Evaluated like any child.
    seed = Program(id=uid.next(), code=seed_code, generation=0, coords=classify(seed_code, backend))
    seed.result = evaluate(seed_code)
    archive.add(seed, 0)
    _log(cfg, f"[it 0] seed score={seed.fitness:.3f} cell={seed.cell} "
              f"{'correct' if seed.result.correct else 'incorrect'}")

    for it in range(1, cfg["max_iters"] + 1):
        # ② staged dimension activation: switch descriptor and re-project the archive
        stages = _maybe_activate_stage(stages, it, cfg, backend, rng, archive, _log)
        B = cfg["branches_per_iteration"]

        if not surrogate:
            # BASELINE path: combined propose+evaluate per branch (identical to the original loop).
            def one_branch(_):
                parent, insp = archive.sample()
                target = pick_target_cell(parent, archive, cfg, rng)
                prompt = build_prompt(task, parent, insp, target, backend)
                code = extract_code(generate(prompt), task["language"])
                child = Program(id=uid.next(), code=code, parent_id=parent.id,
                                generation=parent.generation + 1)
                child.result = evaluate(code)
                child.coords = classify(code, backend)
                return child

            with ThreadPoolExecutor(max_workers=min(B, (os.cpu_count() or 4))) as ex:
                children = list(ex.map(one_branch, range(B)))
        else:
            # ③ SURROGATE path: over-generate k×B cheap candidates, evaluate only the top-B.
            def propose(_):
                parent, insp = archive.sample()
                target = pick_target_cell(parent, archive, cfg, rng)
                prompt = build_prompt(task, parent, insp, target, backend)
                return parent, extract_code(generate(prompt), task["language"])

            n_gen = B * cfg.get("surrogate_overgen", 3)
            with ThreadPoolExecutor(max_workers=min(n_gen, (os.cpu_count() or 4), 16)) as ex:
                proposals = list(ex.map(propose, range(n_gen)))
            keep = set(_rank_candidates([c for _, c in proposals], surrogate, B, rng))
            chosen = [proposals[i] for i in sorted(keep)]
            _log(cfg, f"[it {it}] surrogate kept {len(chosen)}/{n_gen} for evaluation")

            def evaluate_one(pc):
                parent, code = pc
                child = Program(id=uid.next(), code=code, parent_id=parent.id,
                                generation=parent.generation + 1, via_surrogate=True)
                child.result = evaluate(code)
                child.coords = classify(code, backend)
                return child

            with ThreadPoolExecutor(max_workers=min(B, (os.cpu_count() or 4))) as ex:
                children = list(ex.map(evaluate_one, chosen))

        for child in children:
            elite = archive.add(child, it)
            _log(cfg, f"[it {it}] score={child.fitness:.3f} cell={child.cell} "
                      f"static={child.coords} ps={child.result.perf_score} {'*elite*' if elite else ''}")

        # ④b periodic edit trust region: locally refine the best STRUCTURE, staying in-cell
        if edit_fn and cfg.get("edit_refine_every") and it % cfg["edit_refine_every"] == 0:
            _edit_refine_best(archive, task, edit_fn, evaluate, backend, cfg, rng, uid, it, _log)

        # ④a periodic trust-region refinement of the current best (fix structure, tune params)
        if eval_params and cfg.get("refine_every") and it % cfg["refine_every"] == 0:
            _refine_best(archive, eval_params, cfg, rng, uid, it, _log)

        if cfg.get("stop_once_correct") and any(c.result.correct and c.result.compiled for c in children):
            _log(cfg, f"[it {it}] stop_once_correct satisfied")
            break

    # ④b then ④a final polish of the global best (refine structure, then its params)
    if edit_fn and cfg.get("edit_refine_at_end"):
        _edit_refine_best(archive, task, edit_fn, evaluate, backend, cfg, rng, uid, cfg["max_iters"], _log)
    if eval_params and cfg.get("refine_at_end"):
        _refine_best(archive, eval_params, cfg, rng, uid, cfg["max_iters"], _log)

    best = archive.best()
    _log(cfg, f"[done] best score={best.fitness:.3f} cell={best.cell} static={best.coords} "
              f"found@it={best.iteration_found} archive_cells={len(archive.feature_map)}")
    return best, archive


def _make_surrogate(cfg: dict, backend: str):
    s = cfg.get("surrogate", False)
    if not s:
        return None
    return HeuristicSurrogate(backend) if s is True else s   # allow a custom object


def _maybe_activate_stage(stages, it, cfg, backend, rng, archive, logfn):
    """If the earliest pending stage's `until_iter` has passed, switch to the NEXT stage's
    descriptor and re-project. Stages are [{until_iter, descriptor, ...}] in ascending order."""
    if not stages:
        return stages
    cur = stages[0]
    if it > cur.get("until_iter", math.inf):
        stages = stages[1:]
        if stages:
            stage_cfg = {**cfg, **stages[0]}
            archive.reproject(make_descriptor(stage_cfg, backend, rng))
            logfn(cfg, f"[it {it}] stage→ descriptor={stages[0].get('descriptor','static')} "
                       f"re-projected to {len(archive.feature_map)} cells")
    return stages


def _refine_best(archive, eval_params, cfg, rng, uid, it, logfn):
    center = archive.best()
    if not (center.result and center.result.correct):
        return
    logfn(cfg, f"[it {it}] trust-region refine of best (score={center.fitness:.3f}, cell={center.cell})")
    refined = trust_region_refine(center, cfg["param_space"], eval_params, cfg, rng, uid.next,
                                  lambda m: logfn(cfg, m))
    if refined.result and refined.result.score() > center.fitness:
        archive.add(refined, it)
        logfn(cfg, f"[it {it}] refine improved {center.fitness:.3f} → {refined.result.score():.3f} "
                   f"params={getattr(refined, '_best_params', {})}")


def _edit_refine_best(archive, task, edit_fn, evaluate, backend, cfg, rng, uid, it, logfn):
    center = archive.best()
    if not (center.result and center.result.correct):
        return
    logfn(cfg, f"[it {it}] edit trust region on best (score={center.fitness:.3f}, cell={center.coords})")
    refined = edit_trust_region_refine(task, center, edit_fn, evaluate, backend, cfg, rng, uid.next,
                                       lambda m: logfn(cfg, m))
    if refined.id != center.id and refined.fitness > center.fitness:
        archive.add(refined, it)
        logfn(cfg, f"[it {it}] edit-refine improved {center.fitness:.3f} → {refined.fitness:.3f} "
                   f"(stayed in cell {refined.coords})")


DEFAULT_CFG = dict(
    max_iters=20, branches_per_iteration=4, stop_once_correct=False,
    num_islands=4, programs_per_island=10, population_size=1000,
    num_inspirations=2, exploration_ratio=0.2, exploitation_ratio=0.7,
    migration_interval=10, migration_rate=0.1, random_seed=42,
    use_gradient_tracking=True, gradient_sampling_weight=0.3,
    exploration_strategy="mutate", guidance_exploration_rate=0.0,
    gradient_config=dict(fitness_weight=0.4, improvement_rate_weight=0.4,
                         exploration_weight=0.2, max_history=10000),
    verbose=True,
    # ---- advanced convergence strategies: ALL OPT-IN, default OFF (baseline == original loop) ----
    descriptor="static",          # ① 'static' | 'measured' | 'cvt'
    measured_feature_keys=("occupancy", "arithmetic_intensity", "dram_bw_pct"),
    measured_bins=6, cvt_cells=512,
    stages=[],                    # ② e.g. [{'until_iter':8,'descriptor':'static'},{'descriptor':'measured'}]
    surrogate=False,              # ③ True (heuristic) | object with .score(code); top-B of k×B
    surrogate_overgen=3,
    refine_every=None, refine_at_end=False,   # ④a trust-region (needs eval_params + param_space)
    param_space=None, tr_max_evals=16, tr_init_radius=0.4,
    tr_shrink=0.5, tr_grow=1.6, tr_min_radius=0.02,
    edit_refine_every=None, edit_refine_at_end=False,  # ④b edit trust region (needs edit_fn)
    etr_max_evals=12, etr_init_radius=8, etr_min_radius=2, etr_max_radius=40,
    etr_grow=1.6, etr_shrink=0.5,
)


class _ThreadSafeCounter:
    """Monotonic program-id source safe for parallel branches (ThreadPoolExecutor)."""

    def __init__(self):
        import threading
        self._i = 0
        self._lock = threading.Lock()

    def next(self) -> str:
        with self._lock:
            self._i += 1
            return f"prog-{self._i:04d}"


def _log(cfg, msg):
    if cfg.get("verbose", True):
        print(msg)


# ====================================================================================
# Demo world: a toy "kernel" universe so you can run the search with no LLM and no GPU.
# The fake LLM climbs the optimization ladder; the fake evaluator rewards higher levels and
# (for the advanced demo) a hidden continuous tile-size optimum + profiler-like features.
# ====================================================================================
_LADDER = {  # text snippets the classifier recognizes as each (dim, level)
    ("memory_opt", 1): "reinterpret_cast<float4*>", ("memory_opt", 2): "__shared__ ; __syncthreads();",
    ("memory_opt", 3): "#pragma unroll\n acc[i][j] double_buffer",
    ("compute_opt", 1): "fmaf(a,b,c)", ("compute_opt", 2): "running_max online welford",
    ("compute_opt", 3): "flash block attention l_i acc rescale",
    ("parallelism_opt", 1): "tree reduction __shared__", ("parallelism_opt", 2): "__shfl_xor_sync",
    ("parallelism_opt", 3): "warpId laneId warp_tile",
}


def _make_fake_generate(rng, fail_rate=0.12):
    def fake_generate(prompt: str) -> str:
        wanted = {}
        for d in DIMS:
            m = re.search(rf"{d} -> level (\d)", prompt)
            if m:
                wanted[d] = int(m.group(1))
        body = ["// generated kernel"]
        for d in DIMS:
            for L in range(1, wanted.get(d, 0) + 1):
                body.append(_LADDER.get((d, L), ""))
        if rng.random() < fail_rate:                      # occasionally fail → show the gate work
            return "```cuda\n// broken\nint x = ;\n```"
        return "```cuda\n" + "\n".join(body) + "\n__global__ void k(){}\n```"
    return fake_generate


def _make_fake_evaluate(rng, with_features=False):
    def fake_evaluate(code: str) -> EvalResult:
        if "= ;" in code:                                 # "compile error"
            return EvalResult(compiled=False, correct=False, perf_score=1, log="error: expected expression")
        coords = classify(code, "cuda")
        base = 10.0
        runtime = base / (1 + sum(coords)) + rng.uniform(0, 0.3)
        res = EvalResult(compiled=True, correct=True, perf_score=5, runtime=runtime,
                         runtime_improvement=base / runtime,
                         log=f"correct; runtime {runtime:.2f} ms; ~{['memory','compute','balanced'][sum(coords)%3]}-bound")
        if with_features:                                 # profiler-like normalized features (strategy ①)
            res.features = {
                "occupancy": min(1.0, 0.25 + coords[2] * 0.25),
                "arithmetic_intensity": min(1.0, 0.2 + coords[1] * 0.27),
                "dram_bw_pct": min(1.0, 0.3 + coords[0] * 0.23),
            }
        return res
    return fake_evaluate


def _demo():
    """Baseline search — identical to the original loop (all advanced strategies OFF)."""
    rng = random.Random(0)
    task = dict(language="CUDA", reference="def f(x): return x @ x.T  # matmul",
                user_instructions="Optimize the matmul.")
    cfg = dict(max_iters=15, branches_per_iteration=4, verbose=True)
    best, archive = run_evolution(task, _make_fake_generate(rng), _make_fake_evaluate(rng), cfg,
                                  seed_code="// naive\n__global__ void k(){}")
    print("\n=== archive (cell -> best score) ===")
    for cell in sorted(archive.feature_map):
        print(f"  {cell}: {archive.programs[archive.feature_map[cell]].fitness:.3f}")
    print(f"\nbest code:\n{best.code}")


def _demo_advanced():
    """All four convergence strategies ON, to show they compose and the loop stays stable.

    ① staged measured descriptors  ② coarse→fine activation  ③ surrogate pre-ranking
    ④a trust-region refinement of a hidden continuous tile-size optimum."""
    rng = random.Random(0)
    task = dict(language="CUDA", reference="def f(x): return x @ x.T  # matmul",
                user_instructions="Optimize the matmul.")

    # ④ parametric evaluator: fix the elite's STRUCTURE, tune TILE/num_stages toward a hidden optimum.
    OPT = {"tile": 96.0, "stages": 3.4}  # hidden best params

    def eval_params(center: Program, params: Dict[str, float]) -> EvalResult:
        struct = center.result.score()                    # structural score from the archive
        # smooth bowl: closer to OPT → larger speedup bonus on top of the structural score
        pen = sum(((params[k] - OPT[k]) / scale) ** 2 for k, scale in (("tile", 64.0), ("stages", 3.0)))
        bonus = 2.0 * math.exp(-pen)
        runtime = max(0.05, center.result.runtime - bonus * 0.3)
        return EvalResult(compiled=True, correct=True, perf_score=5, runtime=runtime,
                          runtime_improvement=center.result.runtime_improvement + bonus,
                          features=center.result.features,
                          log=f"refined params={params}; +{bonus:.2f} bonus")

    cfg = dict(
        max_iters=15, branches_per_iteration=4, verbose=True,
        # ② coarse static ladder for the first 6 its, then ③①-friendly measured descriptor
        stages=[{"until_iter": 6, "descriptor": "static"}, {"descriptor": "measured"}],
        # ③ over-generate 3× and keep the surrogate's top-B for (expensive) evaluation
        surrogate=True, surrogate_overgen=3,
        # ④a trust-region refine the best every 5 its and once at the end
        refine_every=5, refine_at_end=True,
        param_space={"tile": (16.0, 256.0), "stages": (1.0, 6.0)}, tr_max_evals=18,
    )
    best, archive = run_evolution(task, _make_fake_generate(rng), _make_fake_evaluate(rng, with_features=True),
                                  cfg, seed_code="// naive\n__global__ void k(){}", eval_params=eval_params)
    print("\n=== archive (cell -> best score) ===")
    for cell in sorted(archive.feature_map):
        print(f"  {cell}: {archive.programs[archive.feature_map[cell]].fitness:.3f}")
    print(f"\nbest score={best.fitness:.3f}  best params={getattr(best, '_best_params', 'n/a')}")


def _demo_edit_tr():
    """Strategy ④b in isolation: edit trust region over the LLM diff operator, staying in-cell.

    Every kernel here carries a LOCAL detail `UNROLL=N` that the classifier IGNORES (so editing it
    stays in the same MAP-Elites cell) with a hidden optimum at N=8. To keep the focus on ④b, the
    fake LLM generator always emits ONE fixed structure (so the whole population shares one cell and
    the winner carries the knob). The fake edit_fn emits a mix of:
      * in-cell edits  -- change UNROLL=N            (valid local refinement → may be accepted)
      * out-of-cell    -- inject a higher-level technique (changes the descriptor → REJECTED pre-eval)
      * non-applying   -- a SEARCH not present in the code (malformed patch → shrink)
    Watch the loop reject out-of-cell edits *before* evaluation and converge UNROLL toward 8."""
    rng = random.Random(0)
    task = dict(language="CUDA", reference="def f(x): return x @ x.T  # matmul")

    # one fixed structure (mem L1 + compute L1 + parallelism L2); UNROLL is the local knob.
    def structure(unroll: int) -> str:
        return ("// kernel\nreinterpret_cast<float4*>\nfmaf(a,b,c)\n"
                "__shfl_xor_sync(0xffffffff, v, 16);\n"
                f"#define UNROLL {unroll}\n__global__ void k(){{}}\n")

    seed_code = structure(1)
    cell0 = classify(seed_code, "cuda")                   # the cell ④b must stay inside

    def fixed_generate(prompt: str) -> str:               # ignore targets: always the same structure
        return "```cuda\n" + structure(1) + "```"

    def fake_evaluate(code: str) -> EvalResult:
        if "= ;" in code:
            return EvalResult(compiled=False, correct=False, perf_score=1, log="error: expected expression")
        coords = classify(code, "cuda")
        m = re.search(r"#define UNROLL (\d+)", code)
        unroll = int(m.group(1)) if m else 1
        struct = 10.0 / (1 + sum(coords))                 # structural component (from the cell)
        local = 2.0 * math.exp(-((unroll - 8) / 4.0) ** 2)  # hidden bowl: best at UNROLL=8
        runtime = max(0.05, struct - local * 0.3 + rng.uniform(0, 0.03))
        return EvalResult(compiled=True, correct=True, perf_score=5, runtime=runtime,
                          runtime_improvement=10.0 / runtime,
                          log=f"correct; UNROLL={unroll}; runtime {runtime:.2f} ms")

    def fake_edit(prompt: str) -> str:
        m = re.search(r"#define UNROLL (\d+)", prompt)    # center's code is embedded in the prompt
        cur = int(m.group(1)) if m else 1
        roll = rng.random()
        if roll < 0.22:
            # OUT-OF-CELL: inject a higher-level compute technique → compute axis jumps → REJECTED
            return ("<<<<<<< SEARCH\nfmaf(a,b,c)\n=======\n"
                    "fmaf(a,b,c)\nflash block attention l_i acc rescale\n>>>>>>> REPLACE")
        if roll < 0.34:
            return "<<<<<<< SEARCH\nthis_line_does_not_exist\n=======\nwhatever\n>>>>>>> REPLACE"
        # IN-CELL: nudge UNROLL toward the (unknown-to-it) optimum — a local detail only
        nxt = min(16, max(1, cur + rng.choice([1, 2, 2, 3, -1])))
        return f"<<<<<<< SEARCH\n#define UNROLL {cur}\n=======\n#define UNROLL {nxt}\n>>>>>>> REPLACE"

    cfg = dict(max_iters=6, branches_per_iteration=2, verbose=True,
               edit_refine_every=3, edit_refine_at_end=True,
               etr_max_evals=20, etr_init_radius=8, etr_min_radius=1)
    best, archive = run_evolution(task, fixed_generate, fake_evaluate, cfg,
                                  seed_code=seed_code, edit_fn=fake_edit)
    m = re.search(r"#define UNROLL (\d+)", best.code)
    print(f"\nstayed in cell {cell0}; best score={best.fitness:.3f} cell={best.coords} "
          f"UNROLL={m.group(1) if m else '?'} (hidden optimum=8)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--demo", action="store_true", help="baseline search (advanced strategies OFF)")
    ap.add_argument("--demo-advanced", action="store_true",
                    help="all four convergence strategies ON (① measured/CVT ② staged ③ surrogate ④a trust-region)")
    ap.add_argument("--demo-edit-tr", action="store_true",
                    help="strategy ④b only: edit trust region (SEARCH/REPLACE diffs, reject out-of-cell edits)")
    args = ap.parse_args()
    if args.demo_advanced:
        _demo_advanced()
    elif args.demo_edit_tr:
        _demo_edit_tr()
    elif args.demo:
        _demo()
    else:
        print("Import run_evolution(task, generate, evaluate, cfg, seed_code, eval_params, edit_fn) and "
              "supply your callbacks. Demos: --demo (baseline) | --demo-advanced (①②③④a) | --demo-edit-tr (④b).")
