"""PTL Q3 baseline/head-major/head-major-meta comparison, synthetic selections.

All three use enabled GQA W4/step16, never the old DPAS or scalar comparator.
Each shape checks the unchanged Case FP32 reference, output guards and exact
same-input outputs. --edge-only runs correctness/host-rejection tests, no timing.
Timed samples alternate order; a separate replay of every sample checks exact
outputs without inserting readbacks into the timed no-flush sequence.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json

import numpy as np

import qsa_config
import qsa_harness as H
from qsa_attention_gqa import Q3GQAAttention
from validate_q3_gqa import Case
from validate_q3_split import tensor


VARIANTS = {
    "baseline": {},
    "head-major": {"head_major": True},
    "head-major-meta": {"head_major": True, "vector_meta": True},
}


def implementations(cfg):
    return {name: Q3GQAAttention(cfg, enabled=True, **flags)
            for name, flags in VARIANTS.items()}


class Call:
    """GQA-only call; deliberately does not use the old validator's Call/build."""

    def __init__(self, case, impl):
        self.c, self.impl = case, impl
        self.out = tensor(np.full(case.output_size + 64, np.nan, np.float16))

    def enqueue(self, *, guard=False):
        c = self.c
        self.impl.enqueue(c.inputs, self.out, c.nt, c.batch.num_seqs,
                          c.qbase, c.pitch, c.qhs, c.gd,
                          token_map=c.device_tokens, guard_items=guard)

    def result(self):
        c = self.c
        data = self.out.numpy().reshape(-1)
        assert np.isnan(data[c.output_size:]).all(), "output guard overwritten"
        got = data[:c.output_size].reshape(c.nt, c.hq * c.dv).copy()
        assert np.isfinite(got).all(), "missing/nonfinite output"
        return got


def assert_exact(got, expected, label):
    # Compare FP16 bits, including signed zero, rather than a loose tolerance.
    np.testing.assert_array_equal(got.view(np.uint16), expected.view(np.uint16),
                                  err_msg=label)


def check(case, impls, label):
    from clops import cl

    ref = case.reference()
    calls = {name: Call(case, impl) for name, impl in impls.items()}
    outputs, errors = {}, {}
    for name, call in calls.items():
        call.enqueue(guard=True)
        cl.finish()
        got = call.result()
        np.testing.assert_allclose(got.astype(np.float32), ref, atol=2e-3, rtol=2e-3,
                                   err_msg=f"{label}/{name}/FP32")
        for t in range(case.nt):
            if not len(case.positions(t)):
                np.testing.assert_array_equal(got[t], 0, err_msg="empty attention")
        outputs[name] = got
        errors[name] = float(np.max(np.abs(got.astype(np.float32) - ref), initial=0))
        assert_exact(got, outputs["baseline"], f"{label}/{name}/baseline")
    # Unguarded reverse-order replay also exercises the normal launch geometry.
    replay(calls, outputs, samples=1, label=label)
    print("PASS", json.dumps(dict(case=label, tokens=case.nt, fp32_all_rows=True,
                                  all_outputs_exact=True, errors=errors)), flush=True)
    return calls, outputs


def replay(calls, expected, *, samples, label):
    from clops import cl

    names = tuple(calls)
    for i in range(samples):
        for name in (names[::-1] if i % 2 == 0 else names):
            calls[name].enqueue()
            cl.finish()
            got = calls[name].result()
            assert_exact(got, expected[name], f"{label}/replay{i}/{name}")
            assert_exact(got, expected["baseline"], f"{label}/replay{i}/{name}/baseline")


def expect_value_error(fn, label):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError(f"accepted illegal argument: {label}")


def check_host_rejections(cfg):
    """Invalid constructor options must fail before any compilation/fallback."""
    for flag in ("head_major", "vector_meta", "enabled"):
        for value in (0, 1, None, "true", 1.0, np.bool_(True)):
            for fallback in ("error", "scalar"):
                flags = dict(enabled=True, fallback=fallback)
                flags[flag] = value
                expect_value_error(lambda: Q3GQAAttention(cfg, **flags), f"{flag}={value!r}")
    expect_value_error(lambda: Q3GQAAttention(cfg, fallback="auto"), "fallback")
    for bad in (replace(cfg, heads_num=25), replace(cfg, kv_heads_num=0),
                replace(cfg, heads_num=cfg.kv_heads_num * 17),
                replace(cfg, k_head_size=64), replace(cfg, v_head_size=384),
                replace(cfg, compress_ratio=3), replace(cfg, pa_block_size=15),
                replace(cfg, block_topk=-1), replace(cfg, is_causal=False)):
        expect_value_error(lambda: Q3GQAAttention(bad, enabled=True), "geometry")


def check_runtime_rejections(call):
    c = call.c
    valid = dict(inputs=c.inputs, output=call.out, num_tokens=c.nt,
                 num_seqs=c.batch.num_seqs, q_base=c.qbase, q_pitch=c.pitch,
                 q_head_stride=c.qhs, gate_delta=c.gd, token_map=c.device_tokens)
    invalid = [dict(inputs=[]), dict(inputs=None), dict(output=None),
               dict(num_seqs=0), dict(token_map=None),
               dict(q_base=2), dict(q_pitch=c.pitch + 2),
               dict(q_head_stride=c.qhs + 2), dict(gate_delta=c.gd + 1),
               dict(q_head_stride=16), dict(q_pitch=16),
               dict(q_base=2**32 - 32), dict(num_tokens=2**31)]
    for field in ("num_tokens", "num_seqs", "q_base", "q_pitch", "q_head_stride", "gate_delta"):
        invalid.extend({field: value} for value in (-1, True, 1.5, None, 2**32))
    invalid.extend(dict(guard_items=value) for value in (-1, 1, None, "yes"))
    for overrides in invalid:
        expect_value_error(lambda: call.impl.enqueue(**(valid | overrides)), repr(overrides.keys()))
    call.impl.enqueue([], None, 0, 0, 0, 0, 0, 0)


def edge_cases(cfg):
    mixed = [(0, 0), (0, 1), (1, 3), (3, 5), (14, 7), (15, 17), (31, 19), (63, 1)]
    yield "mixed-pages-pitches", cfg, mixed, {}
    yield "empty-selections-tails", cfg, mixed, dict(pattern="empty")
    for k in (0, 1, 7):
        yield f"K{k}", replace(cfg, block_topk=k), mixed, {}
    for pattern in ("patterns", "overlap", "disjoint"):
        yield pattern, replace(cfg, block_topk=7), [(96, 33)], dict(pattern=pattern)
    for page in (8, 32):
        yield f"page{page}", replace(cfg, pa_block_size=page, block_topk=7), mixed, {}
    for r in (2, 4, 8, 16):
        conf = replace(cfg, compress_ratio=r, block_topk=7)
        seqs = [(64 + rem, 1) for rem in range(r)] + [(33, 17), (0, 0)]
        yield f"r{r}-all-tail-remainders", conf, seqs, {}
    conf = replace(cfg, k_head_size=128, v_head_size=256, scale=128**-0.5, block_topk=7)
    yield "Dh128-Dv256", conf, mixed, {}
    for rep in (1, 3, 6, 12, 16):
        yield f"padded-heads-{rep}", replace(conf, heads_num=rep * cfg.kv_heads_num), mixed, {}
    yield "ungated", replace(cfg, has_output_gate=False, block_topk=7), mixed, {}
    yield "zero-query-second-seed", cfg, mixed, dict(zero_query=True, seed=29)
    yield "unpadded", cfg, mixed, dict(padded=False)
    yield "all-empty", cfg, [(0, 0), (16, 0)], dict(pattern="empty")
    yield "no-sequences", cfg, [], dict(pattern="empty")


def validate_edges(cfg):
    check_host_rejections(cfg)
    compiled = {}
    count = 0
    for label, conf, seqs, options in edge_cases(cfg):
        key = tuple(conf.jit("unused").items())
        if key not in compiled:
            compiled[key] = implementations(conf)
        calls, _ = check(Case(conf, seqs, **options), compiled[key], label)
        if label == "mixed-pages-pitches":
            for call in calls.values():
                check_runtime_rejections(call)
        count += 1
    print(f"ALL {count} GQA VARIANT EDGE CASES + HOST REJECTIONS PASSED", flush=True)


def benchmark(args, cfg):
    from clops import cl

    impls = implementations(cfg)
    names = tuple(impls)
    flusher = H.CacheFlusher()
    for size in args.sizes:
        case = Case(cfg, [(size - args.rows, args.rows)], padded=False)
        calls, expected = check(case, impls, f"size{size}-rows{args.rows}")
        for cache in ("no-flush", "cold96MiB"):
            samples = {name: [] for name in names}
            for i in range(args.warmup + args.samples):
                for name in (names if i % 2 == 0 else names[::-1]):
                    if cache == "cold96MiB" and i >= args.warmup:
                        flusher.flush()
                    calls[name].enqueue()
                    ns = cl.finish()
                    assert len(ns) == 1, ns
                    if i >= args.warmup:
                        samples[name].append(float(ns[0]) * 1e-6)
            for name in names:
                assert_exact(calls[name].result(), expected[name], f"{cache}/{name}/timed-output")
            replay(calls, expected, samples=args.samples, label=f"size{size}/{cache}")
            print("BENCH", json.dumps(dict(size=size, rows=args.rows, cache=cache,
                warmup=args.warmup, samples=args.samples, sample_replays=args.samples,
                all_outputs_exact=True, variants={name: dict(
                    mean_ms=float(np.mean(v)), median_ms=float(np.median(v)), samples_ms=v)
                    for name, v in samples.items()})), flush=True)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sizes", type=int, nargs="+", default=[65536, 131072])
    ap.add_argument("--rows", type=int, default=256)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--samples", type=int, default=10)
    ap.add_argument("--edge-only", action="store_true")
    args = ap.parse_args(argv)
    if (args.warmup < 0 or args.samples < 1 or args.rows < 1 or
            any(size < 1 or size < args.rows for size in args.sizes)):
        ap.error("require warmup >= 0, samples/rows >= 1, and each size >= rows")
    return args


def main():
    args = parse_args()  # Reject illegal CLI arguments before device access.
    from clops import cl

    if "B390" not in str(cl.dev_info()):
        raise RuntimeError("PTL B390 only")
    cl.profiling(True)
    cfg = qsa_config.load()
    if args.edge_only:
        validate_edges(cfg)
    else:
        benchmark(args, cfg)


if __name__ == "__main__":
    main()
