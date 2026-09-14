"""Remote B580 full QSA finalizer A/B on ONE shared fixture/state/scratch/output.

Only finalizer program + workgroup/label changes. JIT, real DPAS Q0/Q1 history,
references, readbacks, and cold96MiB flush stay outside timed enqueue/finish.
All variants are serialized; never use these shared views concurrently.
"""
from __future__ import annotations

import argparse
from copy import copy
from dataclasses import asdict
import gc
import hashlib
import platform
import resource
import sys
from time import perf_counter

import numpy as np

import benchmark_long_context as B
import test_qsa_pipeline as T
from qsa_score_chunked import finalizer_spec


def shared_variant(base, mode):
    """Benchmark-only shallow views: same scorer object and all device tensors."""
    p, q2 = copy(base), copy(base.q2)
    p.q2 = q2
    if mode != "legacy":
        q2.final_name, q2.wg = finalizer_spec(mode, wg=16 if p.phase == "decode" else 8)
        q2.finalizer = T.kernel(p.cfg, q2.final_name,
            QSA_SCORE_ROW_CHUNK=int(not q2.unchunked), QSA_SCORE_WG=16,
            QSA_TOPK_WG=q2.wg, QSA_PARTITION_BLOCKS=q2.partition,
            QSA_DENSE_BYPASS=int(q2.dense_bypass), QSA_DENSE_SCORE_BYPASS=int(q2.dense_bypass))
    p.q2_labels = [base.q2_labels[0], "Q2-topk-" + mode]
    p.labels = base.labels[:3] + p.q2_labels * len(q2.plan.chunks)
    p.labels += [label for label in base.labels if label.startswith("Q3-")]
    assert q2.scorer is base.q2.scorer and q2.scores is base.q2.scores
    assert p.state is base.state and p.out is base.out and q2.tile_map is base.q2.tile_map
    return p


def fingerprint(a):
    # No giant tobytes copy; hash raw bits, including signed zeros and NaN poison.
    return dict(shape=list(a.shape), dtype=a.dtype.str,
                sha256=hashlib.sha256(memoryview(a).cast("B")).hexdigest())


def fingerprints(got):
    return {name: fingerprint(a) for name, a in got.items()}


def check_payload(p, expected, *, mode, when, attention=False):
    """Guards + finite checks from result(); bitwise ALL-payload replay via SHA256."""
    got = p.result()
    actual = fingerprints(got)
    assert actual == expected, (mode, when, [k for k in actual if actual[k] != expected[k]])
    error = None
    if attention:
        rows = B.reference_rows(p)
        error = T.close(got["out"][rows], T.attention_reference(p, got, rows),
                        2e-3, 2e-3, "Q3 own actual selection " + mode)
    B.emit("AB_PASS", dict(variant=mode, when=when, all_payload_bits_exact=True,
        selected_count_exact_rows=p.nt, all_guards=True, full_output_finite_elements=got["out"].size,
        attention_max_abs=error, payloads=actual))


def validate_same_scores(ps, selected, counts):
    """Each chunk scored ONCE; every finalizer consumes that exact device buffer.

    Baseline selection was independently checked for ALL rows by B.validate.
    Verify every output row after each finalizer, score immutability and guards.
    """
    from clops import cl
    base = ps["legacy"]
    inputs = (base.iq.device, base.state.dev("summary"), *base.meta)
    for chunk in base.q2.plan.chunks:
        base.q2.score(chunk, inputs, guard_groups=1)
        cl.finish()
        score_hash = fingerprint(base.q2.read())
        for mode, p in ps.items():
            p.q2.finalize(chunk, p.meta[0], p.meta[1], p.sel.device, p.count.device, guard_groups=1)
            cl.finish()
            np.testing.assert_array_equal(p.sel.read(), selected, err_msg=mode)
            np.testing.assert_array_equal(p.count.read(), counts, err_msg=mode)
            assert fingerprint(p.q2.read()) == score_hash, (mode, "score modified")
        B.emit("SAME_SCORE_CHUNK", dict(begin=chunk.row_begin, end=chunk.row_end,
            variants=list(ps), scores=score_hash, all_selected_counts_exact=True))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", required=True, choices=("prefill", "decode"))
    ap.add_argument("--size", required=True, type=int, choices=(8192, 16384, 32768, 65536))
    ap.add_argument("--variants", nargs="+", choices=("legacy", "fast", "fast-wg8", "fast-wg16"),
                    default=["legacy", "fast", "fast-wg16"])
    ap.add_argument("--cache", nargs="+", choices=("cold96MiB", "no-flush"), default=["cold96MiB"])
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--samples", type=int, default=20)
    args = ap.parse_args()
    if (args.warmup < 0 or args.samples < 1 or len(set(args.variants)) != len(args.variants)
            or args.variants[0] != "legacy" or len(args.variants) < 2):
        ap.error("unique variants starting with legacy; >=2 variants, warmup>=0, samples>=1")
    from clops import cl
    if "B580" not in str(cl.dev_info()):
        raise RuntimeError("Specified remote B580 only")
    cl.profiling(True)
    B.emit("ENV", dict(device=cl.dev_info(), python=sys.version, numpy=np.__version__,
        platform=platform.platform(), argv=sys.argv, sources=B.sources()))
    start = perf_counter()
    cfg = T.qsa_config.load()
    past, q = (0, args.size) if args.phase == "prefill" else (args.size, 1)
    f = T.Fixture(cfg, [past + q])
    state = B.setup_history(f, past)
    base = T.Pipeline(f, [(past, q)], state, optimized=True, phase=args.phase, guard=True,
                      score_options={"finalizer": "legacy"})
    ps = {mode: shared_variant(base, mode) for mode in args.variants}
    cl.finish()
    B.emit("PLAN", dict(phase=args.phase, size=args.size, past=past, q=q, config=asdict(cfg),
        score_plan=asdict(base.q2.plan), event_labels={v: p.labels for v, p in ps.items()},
        shared="ONE fixture/state/scorer/score-scratch/selected/count/output; serial shallow views",
        protocol="v2: cache-matched warmups; no readbacks/reporting between timed samples",
        setup_seconds=perf_counter() - start))
    validated = B.validate(ps["legacy"])
    expected = fingerprints(validated)
    selected, counts = validated["sel"], validated["count"]
    del validated
    gc.collect()
    validate_same_scores(ps, selected, counts)
    del selected, counts
    for mode, p in ps.items():
        p.guard = True
        p.enqueue()
        cl.finish()
        check_payload(p, expected, mode=mode, when="before-timing", attention=True)
        p.guard = False
    flusher = T.H.CacheFlusher() if "cold96MiB" in args.cache else None
    cl.finish()
    variants = list(ps)
    for cache in args.cache:
        for i in range(args.warmup):
            for mode in (variants if i % 2 == 0 else variants[::-1]):
                if cache == "cold96MiB":
                    flusher.flush()
                ps[mode].enqueue()
                cl.finish()
        events, hosts = {v: [] for v in variants}, {v: [] for v in variants}
        for i in range(args.samples):
            for mode in (variants if i % 2 == 0 else variants[::-1]):
                p = ps[mode]
                if cache == "cold96MiB":
                    flusher.flush()
                begin = T.perf_counter_ns()
                p.enqueue()
                ns = cl.finish()
                host_ms = (T.perf_counter_ns() - begin) * 1e-6
                assert len(ns) == len(p.labels), (mode, ns, p.labels)
                event = np.asarray(ns, np.float64) * 1e-6
                events[mode].append(event)
                hosts[mode].append(host_ms)
        # Never interrupt a measurement round with full readbacks (which can
        # cool the GPU for seconds at 64K), references, or logging. Check the
        # actual final shared payload, then replay/check EACH variant below.
        last = variants[-1] if (args.samples - 1) % 2 == 0 else variants[0]
        check_payload(ps[last], expected, mode=last, when="last-measured-" + cache)
        for i in range(args.samples):
            for mode in (variants if i % 2 == 0 else variants[::-1]):
                B.emit("SAMPLE", dict(cache=cache, index=i, variant=mode,
                    event_samples_ms=events[mode][i].tolist(), host_elapsed_ms=hosts[mode][i]))
        for mode, p in ps.items():
            e = np.asarray(events[mode])
            def stage(prefix):
                return T.stats(e[:, [j for j, label in enumerate(p.labels) if label.startswith(prefix)]].sum(axis=1))
            B.emit("BENCH", dict(phase=args.phase, size=args.size, past=past, q=q, variant=mode,
                cache=cache, warmup=args.warmup, samples=args.samples,
                gpu_sum=T.stats(e.sum(axis=1)), host_elapsed=T.stats(hosts[mode]),
                q2_score=stage(p.q2_labels[0]), q2_topk=stage("Q2-topk-"), q3_total=stage("Q3-"),
                stages={name: stage(name) for name in dict.fromkeys(p.labels)},
                event_labels=p.labels, event_samples_ms=e.tolist(), host_samples_ms=hosts[mode],
                score_scratch_bytes=p.q2.plan.allocation_bytes, score_chunks=len(p.q2.plan.chunks),
                max_blocks=p.mb, all_payload_bits_exact=True))
        # Also re-run with extra guard WGs and sampled NumPy after measurements.
        for mode, p in ps.items():
            p.guard = True
            p.enqueue()
            cl.finish()
            check_payload(p, expected, mode=mode, when="after-" + cache, attention=True)
            p.guard = False
    B.emit("DONE", dict(phase=args.phase, size=args.size, elapsed_seconds=perf_counter() - start,
        peak_host_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))


if __name__ == "__main__":
    main()