"""Inputs for the S2 pipeline: ATD arrays in metres, slot teams, Smart Tagging set pieces.

Reuses the cached ATD parse of src/possession/atd.py purely as a loader; nothing else from src/.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, os.path.join(ROOT, "src"), os.path.join(ROOT, "evaluation")):
    if p not in sys.path:
        sys.path.insert(0, p)

from possession import atd  # noqa: E402
from starter.load_smart_tagging import load as load_tags  # noqa: E402

HALVES = [(11185, 90652), (110808, 190198)]  # graded halves (video frames, half-open)
FPS = 25.0


def load():
    raw = atd.load()
    L, W = raw["pitch"]
    P = raw["players"].astype(np.float64)
    B = raw["ball"].astype(np.float64)
    tags = []
    for e in load_tags():
        if (e["code"] or "").strip().lower() == "set pieces":
            team = (e["tags"].get("Team") or "").strip().lower()
            side = 0 if team == "barcelona" else (1 if team in ("atletic club", "athletic club") else -1)
            tags.append((e["start_frame"], e["end_frame"], side, e["tags"].get("Type")))
    return {"px": P[:, :, 0] * L, "py": P[:, :, 1] * W, "pv": P[:, :, 2],
            "bx": B[:, 0] * L, "by": B[:, 1] * W, "teams": raw["teams"],
            "L": L, "W": W, "n": len(B), "set_pieces": tags}
