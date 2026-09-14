"""B580-only, one-shape optimized complete QSA benchmark; no kernel changes.

Uses the actual Fixture/Pipeline and bounded ChunkedQ2. References, history,
readbacks, flush and allocation are outside the timed enqueue/finish slice.
Q1 NumPy projection is streamed; every selected row is checked against exact
own-score top-k, while NumPy scoring and main attention sample explicit rows.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

import numpy as np

import analyze_long_context_roofline as A
import analyze_ptl_roofline as R  # extends A.category() with fast/head-major/vector-meta labels
import test_qsa_pipeline as T

# Peak FP16 compute / memory bandwidth per authorized remote machine (remote_machine.txt).
# Importing R above monkeypatches A.PEAK_FLOPS/A.PEAK_BW to the PTL numbers as a side
# effect; device_peaks()/print_roofline_report() always re-select explicitly per run, so
# that import-time patch is never relied upon here.
BMG_PEAK_FLOPS, BMG_PEAK_BW = 118.784e12, 456e9
PTL_PEAK_FLOPS, PTL_PEAK_BW = 58e12, 110e9


def emit(tag, obj):
    print(tag + " " + json.dumps(obj), flush=True)


def device_peaks(device):
    text = str(device)
    if 'B390' in text:
        return PTL_PEAK_FLOPS, PTL_PEAK_BW
    if 'B580' in text:
        return BMG_PEAK_FLOPS, BMG_PEAK_BW
    raise RuntimeError(f'no known roofline peaks for device {device!r}')


def print_roofline_report(plan, work, bench, device):
    """Human-readable per-kernel measured-vs-roofline report for one (phase, size, cache)
    run: one line per Q0..Q3 kernel stage (measured ms, compute-bound ms, memory-bound ms,
    roofline ms = max(compute, memory)), then a TOTAL line summing Q0..Q3.

    Reuses analyze_long_context_roofline.model_work UNCHANGED (same FLOP/byte accounting
    as the BMG/PTL markdown reports); only the module-level PEAK_FLOPS/PEAK_BW are
    monkeypatched per the device this benchmark is actually running on.
    """
    A.PEAK_FLOPS, A.PEAK_BW = device_peaks(device)
    _global_model, stage_models, _details = A.model_work(plan, work, bench)
    labels = bench['event_labels']
    events = np.asarray(bench['event_samples_ms'])
    stage_measured = {stage: float(np.mean(events[:, [i for i, name in enumerate(labels)
                                                       if R.category(name) == stage]].sum(axis=1)))
                      for stage in R.STAGES}
    print()
    print(f"=== QSA roofline report: {plan['phase']} size={plan['size']} cache={bench['cache']} "
          f"(peak={A.PEAK_FLOPS / 1e12:.3f} TFLOP/s, {A.PEAK_BW / 1e9:.0f} GB/s) ===")
    print(f"{'kernel':<14}{'measured_ms':>14}{'compute_ms':>14}{'memory_ms':>14}{'roofline_ms':>14}{'gap x':>10}")
    total_measured = total_roofline = 0.0
    for stage in R.STAGES:
        measured, model = stage_measured[stage], stage_models[stage]
        total_measured += measured
        total_roofline += model['theory_ms']
        gap = measured / model['theory_ms'] if model['theory_ms'] else float('nan')
        print(f"{stage:<14}{measured:>14.6f}{model['compute_ms']:>14.6f}{model['memory_ms']:>14.6f}"
              f"{model['theory_ms']:>14.6f}{gap:>10.2f}")
    print('-' * 80)
    ratio = total_measured / total_roofline if total_roofline else float('nan')
    print(f"{'TOTAL Q0-Q3':<14}{total_measured:>14.6f}{'':>14}{'':>14}{total_roofline:>14.6f}{ratio:>10.2f}")
    emit('ROOFLINE', dict(phase=plan['phase'], size=plan['size'], cache=bench['cache'],
        peak_flops=A.PEAK_FLOPS, peak_bandwidth=A.PEAK_BW,
        stages={stage: dict(measured_ms=stage_measured[stage], compute_ms=stage_models[stage]['compute_ms'],
                             memory_ms=stage_models[stage]['memory_ms'],
                             roofline_ms=stage_models[stage]['theory_ms']) for stage in R.STAGES},
        total_measured_ms=total_measured, total_roofline_ms=total_roofline, total_gap=ratio))


def reference_rows(p):
    rows = set(map(int, np.linspace(0, p.nt - 1, min(p.nt, 32), dtype=int)))
    for chunk in p.q2.plan.chunks:
        rows.update((chunk.row_begin, chunk.row_end - 1))
        if chunk.row_begin:
            rows.add(chunk.row_begin - 1)
        if chunk.row_end < p.nt:
            rows.add(chunk.row_end)
    # Last dense and first sparse rows, including incomplete r4 tails.
    rows.update(t for t in range(p.nt) if p.batch.seq_of_token(t)[1] in range(2047, 2056))
    return np.asarray(sorted(rows), np.int64)


def validate_prepare_stream(p, got, step=1024):
    """All Q0 cache slots, all Q1 projections/queries/raw keys/summaries.

    These benchmark shapes have B1, aligned past and (q multiple of r or q=1).
    Preserve the original FP32 fresh-key pooling semantics, not f16 raw pooling.
    """
    c, b = p.cfg, p.batch
    assert b.num_seqs == 1 and b.seqs[0][0] % c.compress_ratio == 0
    past, q = b.seqs[0]
    assert step % c.compress_ratio == 0 and (q % c.compress_ratio == 0 or q == 1)
    di, hi, r = c.indexer_head_dim, c.indexer_heads, c.compress_ratio
    errors = dict(projection=0., iq=0., raw=0., summary=0.)
    masks = {name: np.zeros(a.shape[:-1], bool) for name, a in p.before.items()}
    weight = p.f.w.astype(np.float32).T
    for start in range(0, p.nt, step):
        end = min(start + step, p.nt)
        proj = p.host['x'][start:end].astype(np.float32) @ weight
        errors['projection'] = max(errors['projection'], T.close(
            got['projected'][start:end], proj, 3e-5, 2e-5, 'Q1 streamed projection'))
        iq = T.norm_rope(proj[:, :hi * di].reshape(-1, hi, di), p.f.qn,
                         p.pos[start:end, None], p.f)
        errors['iq'] = max(errors['iq'], T.close(got['iq'][start:end], iq,
                                                8e-3, 2e-3, 'Q1 streamed iq'))
        positions = past + np.arange(start, end)
        pages = b.block_indices[positions // c.pa_block_size]
        tips = positions % c.pa_block_size
        for name, source in (('kc', 'key'), ('vc', 'value')):
            np.testing.assert_array_equal(got[name][pages, :, tips], p.host[source][start:end])
            masks[name][pages, :, tips] = True
        raw = proj[:, hi * di:]
        errors['raw'] = max(errors['raw'], T.close(got['raw'][pages, tips],
            raw.astype(np.float16), 8e-3, 2e-3, 'Q1 streamed raw'))
        masks['raw'][pages, tips] = True
        blocks = (end - start) // r
        if blocks:
            pooled = np.zeros((blocks, di), np.float32)
            grouped = raw[:blocks * r].reshape(blocks, r, di)
            for offset in range(r):
                pooled += grouped[:, offset]
            pos = past + start + np.arange(blocks) * r
            pg = b.block_indices[pos // c.pa_block_size]
            slots = (pos // r) % c.summaries_per_page
            summary = T.norm_rope(pooled / np.float32(r), p.f.kn, pos, p.f)
            errors['summary'] = max(errors['summary'], T.close(
                got['summary'][pg, slots], summary, 8e-3, 2e-3, 'Q1 streamed summary'))
            masks['summary'][pg, slots] = True
    for name, mask in masks.items():
        np.testing.assert_array_equal(got[name][~mask], p.before[name][~mask],
                                      err_msg='untouched ' + name)
    return dict(errors=errors, projection_rows=p.nt, iq_rows=p.nt, raw_rows=p.nt,
                summary_rows=int(masks['summary'].sum()), q0_tokens=p.nt,
                reference_chunk_rows=step, untouched_cache_exact=True)


def setup_history(f, past):
    from clops import cl
    start = perf_counter()
    state = T.State(f.empty_state())
    if past:
        # Real external x/K/V, same +3 query RoPE positions and shuffled pages.
        # Q2/Q3 are constructed but NEVER executed here. No scalar Q1.
        p = T.Pipeline(f, [(0, past)], state, optimized=True, guard=True,
                       score_options={'row_cap': 1})
        cl.finish()
        p.prepare()
        ns = cl.finish()
        assert len(ns) == 3, ns
        got = state.snapshot()
        got['iq'], got['projected'] = p.iq.read(), p.projected.read()
        evidence = validate_prepare_stream(p, got)
        emit('HISTORY', dict(tokens=past, source='actual optimized Q0 + DPAS Q1; no Q2/Q3',
                            gpu_stage_ms=dict(zip(p.labels[:3], np.asarray(ns) * 1e-6)),
                            validation=evidence, elapsed_seconds=perf_counter() - start,
                            included_in_timing=False))
        del p, got
        gc.collect()
    return state


def validate(p):
    from clops import cl
    start = perf_counter()
    p.enqueue()
    cl.finish()
    got = p.result()  # all Buffer suffixes, full iq/out finite, split scratch finite
    evidence = validate_prepare_stream(p, got)
    rows = reference_rows(p)
    selected_rows = set(map(int, rows))
    inputs = (p.iq.device, p.state.dev('summary'), *p.meta)
    max_score_error, numpy_score_rows, numpy_selection_differences = 0., 0, []
    np.testing.assert_array_equal(got['count'], np.minimum(p.nc, p.cfg.block_topk))
    for chunk_index, chunk in enumerate(p.q2.plan.chunks):
        before = p.q2.read()
        p.q2.score(chunk, inputs, guard_groups=1)
        cl.finish()
        scores = p.q2.read()
        size = chunk.row_end - chunk.row_begin
        np.testing.assert_array_equal(scores[size:].view(np.uint32), before[size:].view(np.uint32))
        for local, t in enumerate(range(chunk.row_begin, chunk.row_end)):
            nc, count = int(p.nc[t]), int(got['count'][t])
            if p.dense[t]:
                np.testing.assert_array_equal(scores[local].view(np.uint32), before[local].view(np.uint32))
                expected = np.arange(count, dtype=np.int32)
            else:
                assert np.isfinite(scores[local, :nc]).all(), ('nonfinite scores', t)
                expected = T.exact_selection(scores[local], nc, p.cfg.block_topk)
                if t in selected_rows:
                    sr = T.score_reference(p, got, t, t + 1)[0, :nc]
                    max_score_error = max(max_score_error, T.close(
                        scores[local, :nc], sr, 3e-5, 2e-5, 'sampled NumPy score'))
                    numpy_score_rows += 1
                    ref_ids = T.exact_selection(sr, nc, p.cfg.block_topk)
                    if not np.array_equal(expected, ref_ids):
                        numpy_selection_differences.append(dict(row=t,
                            gpu_only=sorted(set(map(int, expected)) - set(map(int, ref_ids))),
                            numpy_only=sorted(set(map(int, ref_ids)) - set(map(int, expected)))))
            np.testing.assert_array_equal(got['sel'][t, :count], expected,
                                          err_msg=f'own-score top-k token {t}')
            np.testing.assert_array_equal(got['sel'][t, count:], -1)
        emit('VALIDATION_CHUNK', dict(index=chunk_index, begin=chunk.row_begin,
                                     end=chunk.row_end, own_score_exact=True))
    ref = T.attention_reference(p, got, rows)
    attention_error = T.close(got['out'][rows], ref, 2e-3, 2e-3, 'Q3 own selection')
    # Real complete replay; compare every output/cache/scratch payload and guard.
    p.enqueue()
    cl.finish()
    for name, value in p.result().items():
        np.testing.assert_array_equal(value, got[name], err_msg='replay ' + name)
    evidence.update(dict(tokens=p.nt, dense_rows=int(p.dense.sum()), sparse_rows=int((~p.dense).sum()),
        metadata_exact_rows=p.nt, own_score_selection_different_rows=0,
        numpy_score_rows=numpy_score_rows, numpy_score_max_abs=max_score_error,
        numpy_selection_differences=numpy_selection_differences,
        attention_ref_rows=len(rows), reference_row_ids=rows.tolist(),
        attention_max_abs=attention_error, full_output_finite_elements=int(got['out'].size),
        full_iq_finite_elements=int(got['iq'].size), all_guards_pass=True,
        unused_scratch_rows_exact=True, replay_all_payloads_exact=True,
        elapsed_seconds=perf_counter() - start))
    emit('PASS', evidence)
    p.guard = False
    return got


def sources():
    root = Path(__file__).resolve().parent
    files = [*root.glob('*.py'), *root.glob('kernels/*'), *root.glob('include/*'),
             root.parent / 'config.json', *root.glob('clops/*.py'), *root.glob('clops/*.cpp')]
    return {str(f.relative_to(root.parent)): hashlib.sha256(f.read_bytes()).hexdigest()
            for f in sorted(files) if f.is_file()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--phase', required=True, choices=('prefill', 'decode'))
    ap.add_argument('--size', required=True, type=int, choices=(1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072))
    ap.add_argument('--warmup', type=int, default=100)
    ap.add_argument('--samples', type=int, default=20)
    ap.add_argument('--topk-finalizer', choices=T.FINALIZER_CHOICES,
                    help='explicit override; omitted preserves existing policy')
    ap.add_argument('--q3-head-major', action='store_true',
                    help='opt-in PTL prefill KV-head-major scheduling and vector metadata')
    ap.add_argument('--cache', nargs='+', choices=('cold96MiB', 'no-flush'), default=['cold96MiB', 'no-flush'])
    args = ap.parse_args()
    if args.warmup < 0 or args.samples < 1:
        ap.error('warmup>=0, samples>=1')
    from clops import cl
    device = cl.dev_info()
    # Only the two remote machines named in remote_machine.txt are authorized:
    # the Arc B580 (BMG) and the PTL Arc B390 iGPU.
    if not any(name in str(device) for name in ('B580', 'B390')):
        raise RuntimeError('Run only on a specified remote machine (Arc B580 or PTL Arc B390)')
    cl.profiling(True)
    emit('ENV', dict(device=device, python=sys.version, numpy=np.__version__,
                     platform=platform.platform(), argv=sys.argv, sources=sources()))
    begin = perf_counter()
    cfg = T.qsa_config.load()
    past, q = (0, args.size) if args.phase == 'prefill' else (args.size, 1)
    f = T.Fixture(cfg, [past + q])
    state = setup_history(f, past)
    p = T.Pipeline(f, [(past, q)], state, optimized=True, phase=args.phase, guard=True,
                   score_options={'finalizer': args.topk_finalizer},
                   attention_options=({'head_major': True, 'vector_meta': True}
                                      if args.q3_head_major else None))
    plan_record = dict(phase=args.phase, size=args.size, past=past, q=q, config=asdict(cfg),
                       score_plan=asdict(p.q2.plan), event_labels=p.labels,
                       setup_seconds=perf_counter() - begin)
    emit('PLAN', plan_record)
    validated = validate(p)
    flusher = T.H.CacheFlusher() if 'cold96MiB' in args.cache else None
    nc = p.nc.astype(np.int64)
    positions = np.asarray([p.batch.seq_of_token(t)[1] for t in range(p.nt)], np.int64)
    counts = np.minimum(nc, cfg.block_topk)
    work_record = dict(phase=args.phase, size=args.size, past=past, q=q,
        n_complete_sum=int(nc.sum()), sparse_candidates_sum=int(nc[~p.dense].sum()),
        max_complete=int(nc.max()), dense_rows=int(p.dense.sum()), sparse_rows=int((~p.dense).sum()),
        summaries_produced=(past + q) // cfg.compress_ratio - past // cfg.compress_ratio,
        selected_blocks_sum=int(counts.sum()), tail_positions_sum=int(((positions + 1) % cfg.compress_ratio).sum()),
        attended_positions_sum=int((counts * cfg.compress_ratio + (positions + 1) % cfg.compress_ratio).sum()),
        projection_flops=2 * q * cfg.hidden_size * (cfg.indexer_heads + 1) * cfg.indexer_head_dim)
    emit('WORK', work_record)
    for cache in args.cache:
        cl.finish()
        for _ in range(args.warmup):
            p.enqueue()
            cl.finish()
        events, hosts = [], []
        for _ in range(args.samples):
            if cache == 'cold96MiB':
                flusher.flush()
            start = T.perf_counter_ns()
            p.enqueue()
            ns = cl.finish()
            hosts.append((T.perf_counter_ns() - start) * 1e-6)
            assert len(ns) == len(p.labels), (ns, p.labels)
            events.append(np.asarray(ns, np.float64) * 1e-6)
        e = np.asarray(events)
        stages = {name: T.stats(e[:, [i for i, label in enumerate(p.labels) if label == name]].sum(axis=1))
                  for name in dict.fromkeys(p.labels)}
        for name, value in p.result().items():
            np.testing.assert_array_equal(value, validated[name], err_msg='post-benchmark ' + name)
        bench_record = dict(phase=args.phase, size=args.size, past=past, q=q, cache=cache,
            warmup=args.warmup, samples=args.samples, gpu_sum=T.stats(e.sum(axis=1)),
            host_elapsed=T.stats(hosts), stages=stages,
            q3_total=T.stats(e[:, [i for i, label in enumerate(p.labels) if label.startswith('Q3-')]].sum(axis=1)),
            event_labels=p.labels, event_samples_ms=e.tolist(), host_samples_ms=hosts,
            score_scratch_bytes=p.q2.plan.allocation_bytes, score_payload_bytes=p.q2.plan.scratch_bytes,
            score_chunks=len(p.q2.plan.chunks), score_rows=p.q2.plan.scratch_rows,
            max_blocks=p.mb, projection_scratch_bytes=p.projected.size * 4,
            split_scratch_bytes=(p.acc.size + p.ml.size) * 4,
            post_benchmark_all_payloads_exact=True)
        emit('BENCH', bench_record)
        print_roofline_report(plan_record, work_record, bench_record, device)
    emit('DONE', dict(phase=args.phase, size=args.size, elapsed_seconds=perf_counter() - begin))


if __name__ == '__main__':
    main()