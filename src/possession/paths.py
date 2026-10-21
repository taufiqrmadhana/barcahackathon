"""Repository paths, and the upstream modules (games.py, starter/, evaluation/) on sys.path."""

import os
import sys

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(SRC)
CACHE_DIR = os.environ.get("BARCA_CACHE_DIR") or os.path.join(ROOT, "cache")
RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")

for path in (ROOT, os.path.join(ROOT, "evaluation")):
    if path not in sys.path:
        sys.path.insert(0, path)
