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

You supply two callbacks:
    generate(prompt:str) -> str          # call your LLM; return its raw answer (code or fenced code)
    evaluate(code:str)  -> EvalResult     # compile + test + benchmark; return the result

Run the built-in demo (no LLM/GPU needed) to watch the search converge:
    python evolve_loop.py --demo

Wire it to a real backend by importing run_evolution() and passing your two callbacks.
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
    coords: Tuple[int, ...] = ()
    result: Optional[EvalResult] = None
    island: int = 0

    @property
    def fitness(self) -> float:
        return self.result.score() if self.result else -math.inf


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
    def __init__(self, cfg: dict, rng: random.Random):
        self.cfg = cfg
        self.rng = rng
        self.levels = 4
        self.programs: Dict[str, Program] = {}
        self.feature_map: Dict[Tuple[int, ...], str] = {}     # cell -> elite id
        self.num_islands = cfg.get("num_islands", 4)
        self.islands: List[set] = [set() for _ in range(self.num_islands)]
        self.active_island = 0
        self._since_switch = 0
        self.generation = 0
        self.gradient = QDGradient(cfg.get("gradient_config", {})) if cfg.get("use_gradient_tracking", True) else None

    # ---- insert -------------------------------------------------------------------
    def add(self, prog: Program, iteration: int) -> bool:
        prog.iteration_found = iteration
        prog.island = self.active_island
        self.programs[prog.id] = prog
        self.islands[self.active_island].add(prog.id)
        became_elite = False
        cell = prog.coords
        cur = self.feature_map.get(cell)
        if cur is None or prog.fitness > self.programs[cur].fitness:
            self.feature_map[cell] = prog.id
            became_elite = True
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
# The controller loop (controller.py::_run_single)
# ====================================================================================
def run_evolution(task: dict,
                  generate: Callable[[str], str],
                  evaluate: Callable[[str], EvalResult],
                  cfg: Optional[dict] = None,
                  seed_code: str = "") -> Tuple[Program, Archive]:
    cfg = {**DEFAULT_CFG, **(task.get("cfg") or {}), **(cfg or {})}
    rng = random.Random(cfg.get("random_seed", 42))
    backend = task["language"].lower().replace("c++", "cpp")
    backend = {"ocl": "opencl"}.get(backend, backend)
    archive = Archive(cfg, rng)
    _uid = _counter()

    # iteration 0: seed (reference scaffold / provided best). Evaluated like any child.
    seed = Program(id=next(_uid), code=seed_code, generation=0, coords=classify(seed_code, backend))
    seed.result = evaluate(seed_code)
    seed.coords = classify(seed_code, backend)
    archive.add(seed, 0)
    _log(cfg, f"[it 0] seed score={seed.fitness:.3f} cell={seed.coords} "
              f"{'correct' if seed.result.correct else 'incorrect'}")

    for it in range(1, cfg["max_iters"] + 1):
        def one_branch(_):
            parent, insp = archive.sample()
            target = pick_target_cell(parent, archive, cfg, rng)
            prompt = build_prompt(task, parent, insp, target, backend)
            code = extract_code(generate(prompt), task["language"])
            child = Program(id=next(_uid), code=code, parent_id=parent.id,
                            generation=parent.generation + 1)
            child.result = evaluate(code)
            child.coords = classify(code, backend)
            return child

        workers = min(cfg["branches_per_iteration"], (os.cpu_count() or 4))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            children = list(ex.map(one_branch, range(cfg["branches_per_iteration"])))

        for child in children:
            elite = archive.add(child, it)
            mark = "*elite*" if elite else ""
            _log(cfg, f"[it {it}] score={child.fitness:.3f} cell={child.coords} "
                      f"ps={child.result.perf_score} {mark}")

        if cfg.get("stop_once_correct") and any(c.result.correct and c.result.compiled for c in children):
            _log(cfg, f"[it {it}] stop_once_correct satisfied")
            break

    best = archive.best()
    _log(cfg, f"[done] best score={best.fitness:.3f} cell={best.coords} "
              f"found@it={best.iteration_found} archive_cells={len(archive.feature_map)}")
    return best, archive


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
)


def _counter():
    i = 0
    while True:
        i += 1
        yield f"prog-{i:04d}"


def _log(cfg, msg):
    if cfg.get("verbose", True):
        print(msg)


# ====================================================================================
# Demo: a toy "kernel" world so you can run the search with no LLM and no GPU.
# The fake LLM climbs the optimization ladder; the fake evaluator rewards higher levels.
# ====================================================================================
def _demo():
    rng = random.Random(0)
    LADDER = {  # text snippets that the classifier recognizes as each (dim, level)
        ("memory_opt", 1): "reinterpret_cast<float4*>", ("memory_opt", 2): "__shared__ ; __syncthreads();",
        ("memory_opt", 3): "#pragma unroll\n acc[i][j] double_buffer",
        ("compute_opt", 1): "fmaf(a,b,c)", ("compute_opt", 2): "running_max online welford",
        ("compute_opt", 3): "flash block attention l_i acc rescale",
        ("parallelism_opt", 1): "tree reduction __shared__", ("parallelism_opt", 2): "__shfl_xor_sync",
        ("parallelism_opt", 3): "warpId laneId warp_tile",
    }

    def fake_generate(prompt: str) -> str:
        # read the demanded target levels out of the prompt and emit matching snippets
        wanted = {}
        for d in DIMS:
            m = re.search(rf"{d} -> level (\d)", prompt)
            if m:
                wanted[d] = int(m.group(1))
        body = ["// generated kernel"]
        for d in DIMS:
            lvl = wanted.get(d, 0)
            for L in range(1, lvl + 1):
                body.append(LADDER.get((d, L), ""))
        # occasionally "fail" to show the gate working
        if rng.random() < 0.12:
            return "```cuda\n// broken\nint x = ;\n```"
        return "```cuda\n" + "\n".join(body) + "\n__global__ void k(){}\n```"

    def fake_evaluate(code: str) -> EvalResult:
        if "= ;" in code:  # "compile error"
            return EvalResult(compiled=False, correct=False, perf_score=1, log="error: expected expression")
        coords = classify(code, "cuda")
        # higher optimization levels -> lower (better) runtime -> higher speedup
        base = 10.0
        runtime = base / (1 + sum(coords)) + rng.uniform(0, 0.3)
        speedup = base / runtime
        return EvalResult(compiled=True, correct=True, perf_score=5,
                          runtime=runtime, runtime_improvement=speedup,
                          log=f"correct; runtime {runtime:.2f} ms; ~{['memory','compute','balanced'][sum(coords)%3]}-bound")

    task = dict(language="CUDA", reference="def f(x): return x @ x.T  # matmul",
                user_instructions="Optimize the matmul.")
    cfg = dict(max_iters=15, branches_per_iteration=4, verbose=True)
    best, archive = run_evolution(task, fake_generate, fake_evaluate, cfg, seed_code="// naive\n__global__ void k(){}")
    print("\n=== archive (cell -> best score) ===")
    for cell in sorted(archive.feature_map):
        print(f"  {cell}: {archive.programs[archive.feature_map[cell]].fitness:.3f}")
    print(f"\nbest code:\n{best.code}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--demo", action="store_true", help="run the no-LLM/no-GPU demo search")
    args = ap.parse_args()
    if args.demo:
        _demo()
    else:
        print("Import run_evolution(task, generate, evaluate, cfg, seed_code) and supply your "
              "LLM + evaluator callbacks. Run with --demo to see the search work end-to-end.")
