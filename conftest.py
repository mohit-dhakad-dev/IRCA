"""Make the repo root importable so `tools`, `agent`, and `main` resolve in tests.

Without this, bare `pytest` fails to collect: unlike `python -m pytest`, the
`pytest` console script does not put the current directory on sys.path, so
top-level packages are invisible. Keeping this at the repo root means both
invocations behave identically.
"""

import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
