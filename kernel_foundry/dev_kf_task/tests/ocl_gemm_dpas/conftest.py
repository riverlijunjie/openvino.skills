"""
Do not modify this file.
This file configures pytest and imports fixtures to simplify the development of task.py
"""

from pathlib import Path
import sys

# Allow running pytest directly from workspace tasks without installing kernelfoundry.
_repo_root = Path(__file__).resolve().parents[2]
_kernelfoundry_pkg = _repo_root / "kernelfoundry.internal" / "kernelfoundry"
if _kernelfoundry_pkg.exists():
	sys.path.insert(0, str(_kernelfoundry_pkg))

from kernelfoundry.conftest import *
