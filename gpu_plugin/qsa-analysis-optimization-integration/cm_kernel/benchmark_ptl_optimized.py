"""Same-state alternating PTL baseline/optimized full-pipeline benchmark.

Baseline prefill: fast-wg16 + token-major GQA. Baseline decode: legacy WG16.
New: cached WG16; prefill also head-major/vector-metadata GQA. Original default
policies remain unchanged. Each variant uses the same inputs/state and validates
all own-score top-k rows, sampled FP32 attention, full output equality and guards.
"""
from __future__ import annotations
import argparse
import gc
import json
import platform
import sys
from time import perf_counter_ns
import numpy as np
import benchmark_long_context as B
import test_qsa_pipeline as T


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--phase', choices=('prefill', 'decode'), required=True)
    ap.add_argument('--size', type=int, choices=(4096, 65536, 131072), required=True)
    ap.add_argument('--warmup', type=int, default=20)
    ap.add_argument('--samples', type=int, default=20)
    ap.add_argument('--validate-only', action='store_true',
                    help='all-row selection and full bytewise A/B validation; no timing')
    args = ap.parse_args()
    if args.warmup < 0 or args.samples < 1:
        ap.error('warmup>=0, samples>=1')
    from clops import cl
    device = cl.dev_info()
    if 'B390' not in str(device):
        raise RuntimeError('specified remote PTL B390 only')
    cl.profiling(True)
    # Avoid enormous tile-map/source/event arrays in console; source hashes still
    # retained in ENV and per-stage sample arrays in BENCH.
    B.emit('ENV', dict(device=device, python=sys.version, platform=platform.platform(),
                       argv=sys.argv, sources=B.sources()))
    cfg = T.qsa_config.load()
    past, q = (0, args.size) if args.phase == 'prefill' else (args.size, 1)
    f = T.Fixture(cfg, [past + q])
    state = B.setup_history(f, past)
    p = T.Pipeline(f, [(past, q)], state, optimized=True, phase=args.phase, guard=True,
        score_options={'finalizer': 'fast-wg16' if args.phase == 'prefill' else 'cooperative'})
    baseline_q2, baseline_q3, baseline_labels = p.q2, p.q3, p.labels.copy()
    B.emit('PLAN', dict(phase=args.phase, size=args.size, rows=p.nt,
        score_chunks=len(p.q2.plan.chunks), scratch_rows=p.q2.plan.scratch_rows))
    baseline = B.validate(p)
    # Construct once outside timing; same scorer inputs/outputs and same scratch
    # shape. Only score-scratch buffers differ, initialized identically.
    from qsa_score_chunked import ChunkedQ2
    from qsa_attention_gqa import Q3GQAAttention
    optimized_q2 = ChunkedQ2(cfg, p.batch, p.mb,
        backend='dpas' if args.phase == 'prefill' else 'partition',
        partition_blocks=p.partition, finalizer='fast-wg16-cached')
    optimized_q3 = (Q3GQAAttention(cfg, enabled=True, head_major=True, vector_meta=True)
                    if args.phase == 'prefill' else baseline_q3)
    optimized_labels = [s.replace('Q2-fast-WG16', 'Q2-fast-WG16-cached').replace(
        'Q2-WG16', 'Q2-fast-WG16-cached').replace('Q3-gqa-W4', 'Q3-gqa-W4-head-major-vector-meta')
        for s in baseline_labels]
    variants = {'baseline': (baseline_q2, baseline_q3, baseline_labels),
                'optimized': (optimized_q2, optimized_q3, optimized_labels)}
    def select(name):
        p.q2, p.q3, p.labels = variants[name]
    select('optimized')
    p.guard = True
    optimized = B.validate(p)
    for name in baseline:
        np.testing.assert_array_equal(optimized[name].view(np.uint8), baseline[name].view(np.uint8),
                                      err_msg='A/B full payload bits ' + name)
    B.emit('AB_PASS', dict(full_payloads_bit_exact=True, all_topk_rows_exact=True,
                          size=args.size, phase=args.phase))
    if args.validate_only:
        B.emit('VALIDATION_DONE', dict(size=args.size, phase=args.phase, full_bytewise_ab=True))
        return
    del baseline, optimized
    gc.collect()
    flush = T.H.CacheFlusher()
    for cache in ('cold96MiB', 'no-flush'):
        events = {name: [] for name in variants}
        hosts = {name: [] for name in variants}
        for i in range(args.warmup + args.samples):
            for name in (tuple(variants) if i % 2 == 0 else tuple(variants)[::-1]):
                select(name)
                if cache == 'cold96MiB' and i >= args.warmup:
                    flush.flush()
                start = perf_counter_ns()
                p.enqueue()
                ns = cl.finish()
                host = (perf_counter_ns() - start) * 1e-6
                assert len(ns) == len(p.labels), (len(ns), len(p.labels))
                if i >= args.warmup:
                    events[name].append(np.asarray(ns, np.float64) * 1e-6)
                    hosts[name].append(host)
        results = {}
        for name in variants:
            select(name)
            e = np.asarray(events[name])
            stages = {label: e[:, [i for i, s in enumerate(p.labels) if s == label]].sum(axis=1)
                      for label in dict.fromkeys(p.labels)}
            results[name] = dict(total=T.stats(e.sum(axis=1)), host=T.stats(hosts[name]),
                stages={k: dict(**T.stats(v), samples_ms=v.tolist()) for k, v in stages.items()},
                samples_ms=e.sum(axis=1).tolist())
        # Check every payload and both score-scratch guards outside timing.
        select('baseline')
        p.enqueue(); cl.finish()
        want = p.result()
        select('optimized')
        p.enqueue(); cl.finish()
        got = p.result()
        for key in want:
            np.testing.assert_array_equal(got[key].view(np.uint8), want[key].view(np.uint8),
                                          err_msg='post-timing A/B bits ' + key)
        del want, got
        B.emit('BENCH', dict(phase=args.phase, size=args.size, cache=cache,
            warmup=args.warmup, samples=args.samples, variants=results,
            speedup=results['baseline']['total']['mean_ms'] / results['optimized']['total']['mean_ms'],
            post_timing_ab_exact=True))
    B.emit('DONE', dict(phase=args.phase, size=args.size))


if __name__ == '__main__':
    main()
