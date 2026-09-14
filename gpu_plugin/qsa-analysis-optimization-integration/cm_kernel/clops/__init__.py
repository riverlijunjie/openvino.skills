"""Minimal clops package: OpenCL + CM kernel support only.

Only the OpenCL and CM components are included.
Import it with `from libs.clops import cl`; the C++ extension is built on first import.
"""

from . import cl
