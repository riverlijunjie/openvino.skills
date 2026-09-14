"""Reproduce the unchanged historical roofline audit against its exact sources.

The three evolving Python sources are restored IN MEMORY by reversing only
known explicit-finalizer additions, then MUST match original ENV SHA256 bytes.
All 56 recorded sources are copied to a temporary source-root (never overwrite
the working tree). Original logs, analyzer, tests and reports remain unchanged.
This is a hash-checked reverse-delta archive, not permission to ignore drift.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import analyze_long_context_roofline as A


def restore(name, text):
    if name == "cm_kernel/benchmark_long_context.py":
        text = text.replace("    ap.add_argument('--topk-finalizer', choices=T.FINALIZER_CHOICES,\n"
                            "                    help='explicit override; omitted preserves existing policy')\n", "")
        return text.replace("phase=args.phase, guard=True,\n"
                            "                   score_options={'finalizer': args.topk_finalizer})",
                            "phase=args.phase, guard=True)")
    if name == "cm_kernel/qsa_score_chunked.py":
        begin, end = text.index("FINALIZER_CHOICES ="), text.index("@dataclass")
        text = text[:begin] + "\n\n" + text[end:]
        text = text.replace("unchunked=False, finalizer=None):", "unchunked=False):")
        begin = text.index("        final_name, final_wg =")
        end = text.index("        if cfg.block_topk", begin)
        text = text[:begin] + text[end:]
        text = text.replace("self.wg, self.dense_bypass = final_wg, dense_bypass",
                            "self.wg, self.dense_bypass = (wg if cooperative else 1), dense_bypass")
        return text.replace("self.final_name = final_name",
                            'self.final_name = "qsa_topk_cooperative" if cooperative else "qsa_topk_finalization"')
    if name == "cm_kernel/test_qsa_pipeline.py":
        text = text.replace("ChunkedQ2, DEFAULT_SCORE_BYTES, FINALIZER_CHOICES", "ChunkedQ2, DEFAULT_SCORE_BYTES")
        begin = text.index("        if self.q2.final_name ==")
        end = text.index("        self.labels +=", begin)
        text = text[:begin] + text[end:]
        begin = text.index('    ap.add_argument("--topk-finalizer"')
        end = text.index('    ap.add_argument("--mode"', begin)
        text = text[:begin] + text[end:]
        return text.replace("row_cap=args.score_row_cap, unchunked=args.score_unchunked,\n"
                            "                         finalizer=args.topk_finalizer)",
                            "row_cap=args.score_row_cap, unchunked=args.score_unchunked)")
    return text


@contextmanager
def historical_root():
    records = A.read_records(A.HERE / "long_context_logs/prefill_1024.log")
    with tempfile.TemporaryDirectory(prefix="qsa-historical-sources-") as directory:
        root = Path(directory)
        for name, digest in records["ENV"][0]["sources"].items():
            source = (A.HERE.parent / name).resolve()
            A.require(source.is_relative_to(A.HERE.parent.resolve()), "invalid source path")
            data = source.read_bytes()
            if hashlib.sha256(data).hexdigest() != digest:
                data = restore(name, data.decode()).encode()
            A.require(hashlib.sha256(data).hexdigest() == digest, "historical restoration mismatch: " + name)
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        A.audit_sources(records, root)
        yield root


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--test", action="store_true", help="run the unchanged 12 historical tests against verified source snapshot")
    args = ap.parse_args()
    with historical_root() as root:
        if args.test:
            original = A.audit_sources
            with patch.object(A, "audit_sources", side_effect=lambda records, _root: original(records, root)):
                suite = unittest.defaultTestLoader.loadTestsFromName("test_analyze_long_context_roofline")
                result = unittest.TextTestRunner(verbosity=2).run(suite)
                if not result.wasSuccessful():
                    raise SystemExit(1)
        else:
            data = A.analyze(A.HERE / "long_context_logs", root, A.HERE / "LONG_CONTEXT_PERFORMANCE_CN.md")
            A.require(A.render_report(data) == (A.HERE / "LONG_CONTEXT_ROOFLINE_CN.md").read_text(),
                      "historical report drift")
            print("PASS: exact historical source hashes, 28 measurements/6960 events, unchanged roofline report")


if __name__ == "__main__":
    main()