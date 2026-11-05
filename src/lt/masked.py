"""Hidden-ball comparison on the public window: ATD only vs ATD + video, same masks.

The ATD ball is hidden with the real missing-ball mask of another stretch of the match
(transplant, as in robustness.py). The video arm may use pass-1 detections (crops centred on the
ATD ball) only on frames whose ATD ball is still visible, and the ATD-independent grid detections
everywhere. The shape model's probabilities are frozen from the unmasked run for both arms, so the
comparison is between ball tracks, not an estimate of absolute hidden-ball performance.
"""

import os
import sys
from dataclasses import replace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE)]
from lt.harness import *  # noqa: F401,F403,E402
from lt import harness as hz  # noqa: E402
from cv import balltrack  # noqa: E402
from games import GAME, GAMES  # noqa: E402
from starter.load_homography import load as load_homography  # noqa: E402

PUBLIC = (11185, 18685)
PUB = [(1, 11185, 18684)]


def transplant(ball, graded, rng):
    n = PUBLIC[1] - PUBLIC[0]
    g = graded.copy()
    g[PUBLIC[0] - n:PUBLIC[1]] = False
    starts = [s for s in np.flatnonzero(g) if g[s:s + n].all()]
    src = int(rng.choice(starts[::25]))
    mask = np.isnan(ball[src:src + n, 0])
    out = ball.copy()
    hidden = np.zeros(len(ball), bool)
    hidden[PUBLIC[0]:PUBLIC[1]] = mask & ~np.isnan(ball[PUBLIC[0]:PUBLIC[1], 0])
    out[PUBLIC[0]:PUBLIC[1]][mask] = np.nan
    return out, hidden, src


class Arms:
    def __init__(self, grid_tags="grid_pub+grid_pub_rest", prm=balltrack.Params):
        self.inputs = hz.pipeline.Inputs()
        self.H = load_homography()[0]
        self.pass1 = balltrack.load_candidates("atd")
        self.grid = balltrack.load_candidates(grid_tags)
        self.prm = prm
        self.graded = self.inputs.graded

    def fused(self, ball, hidden, use_pass1=True):
        cands = {}
        for f in range(*PUBLIC):
            parts = []
            if use_pass1 and not hidden[f] and f in self.pass1:
                parts.append(self.pass1[f])
            if f in self.grid:
                parts.append(self.grid[f])
            parts = [p for p in parts if len(p)]
            if parts:
                c = np.concatenate(parts)
                cands[f] = c[np.argsort(-c[:, 4])]
        atd = self.inputs.atd
        return balltrack.fuse(ball[:, :2], cands, self.H, [(1, PUBLIC[0], PUBLIC[1] - 1)],
                              GAMES[GAME]["pitch"], self.prm, players=atd["players"])


def write_ball(name, xy):
    np.savez(os.path.join(hz.ROOT, "cache", "cv", name + ".npz"), xy=xy)
