#!/usr/bin/env python3
"""Compatibility wrapper for the generic shared report builder."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "utils"))

from build_model_report import main  # type: ignore


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    model_name = here.name
    extra = [
        "--model-dir", str(ROOT / "models" / model_name),
        "--output-dir", str(here),
    ]
    raise SystemExit(main(extra + sys.argv[1:]))
