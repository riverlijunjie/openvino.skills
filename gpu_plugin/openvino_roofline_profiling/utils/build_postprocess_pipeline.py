#!/usr/bin/env python3
"""Unified post-parse pipeline for roofline profiling artifacts.

Given parsed metrics under outputs/<model>/, generate:
- performance_metrics_<platform>.json via the shared report engine
- kernel_tables.md via the shared kernel-table engine
- optional db/metrics.db update via utils/build_db.py
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from build_model_report import main as build_report_main
from build_kernel_tables import main as build_kernel_tables_main

SKILL_ROOT = Path(__file__).resolve().parent.parent


def detect_platforms(output_dir: Path) -> list[str]:
    result = []
    for candidate in ["BMG", "PTL", "LNL"]:
        if (output_dir / f"{candidate.lower()}_metrics.json").exists():
            result.append(candidate)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--platform", action="append", dest="platforms")
    parser.add_argument("--kernel-tables-out", type=Path, default=None)
    parser.add_argument("--rebuild-db", action="store_true")
    args = parser.parse_args(argv)

    model_name = args.model_dir.name
    platforms = args.platforms or detect_platforms(args.output_dir)
    if not platforms:
        raise SystemExit("No parsed metrics found; expected <platform>_metrics.json under output-dir")

    for platform in platforms:
        report_out = args.output_dir / f"performance_metrics_{platform.lower()}.json"
        build_report_main([
            "--model-dir", str(args.model_dir),
            "--output-dir", str(args.output_dir),
            "--platform", platform,
            "--out", str(report_out),
        ])

    kernel_out = args.kernel_tables_out or (args.output_dir / "kernel_tables.md")
    kernel_args = [
        "--model-dir", str(args.model_dir),
        "--output-dir", str(args.output_dir),
        "--out", str(kernel_out),
    ]
    for platform in platforms:
        kernel_args.extend(["--platform", platform])
    build_kernel_tables_main(kernel_args)

    if args.rebuild_db:
        subprocess.run([
            sys.executable,
            str(SKILL_ROOT / "utils" / "build_db.py"),
            "--model", model_name,
        ], check=True)

    print(f"Generated reports for {model_name}: {', '.join(platforms)}")
    print(f"Kernel tables: {kernel_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
