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


WINDOW = 25000


def windowed(inputs, n=WINDOW):
    """Cut every per-frame array to the first n frames (the public window sits inside), so the
    full-match features (ownership, evidence) are not recomputed for a five-minute test."""
    full = inputs.n
    inputs.atd = {k: (v[:n] if hasattr(v, "shape") and len(v) == full else v)
                  for k, v in inputs.atd.items()}
    inputs.tags = {k: (v[:n] if hasattr(v, "shape") and len(v) == full else v)
                   for k, v in inputs.tags.items()}
    inputs.graded = inputs.graded[:n].copy()
    inputs.graded[18685:] = False
    inputs.blocks = [(1, PUBLIC[0], PUBLIC[1] - 1)]
    inputs.n = n
    inputs._cache = {}
    return inputs


class Arms:
    def __init__(self, grid_tags="grid_pub+grid_pub_rest", prm=balltrack.Params, grid_floor=None):
        self.inputs = hz.pipeline.Inputs()
        self.full_n = self.inputs.n
        self.grid_floor = grid_floor
        self.H = load_homography()[0]
        self.pass1 = balltrack.load_candidates("atd")
        from cv.build_fused import GRID_MIN_SCORE
        floor = GRID_MIN_SCORE if grid_floor is None else grid_floor
        self.grid = balltrack.load_candidates(grid_tags, {t: floor for t in grid_tags.split("+")})
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


class Bench:
    """Score decoder configs on the clean public window and on transplanted hidden-ball masks.

    prepare() writes one ATD-only and one ATD+video ball file per seed (cache/cv/mask_*); run()
    decodes a config on every seed with the shape model frozen and returns per-seed scores."""

    def __init__(self, seeds=(-1, 0, 1, 2, 3, 4, 5), prefix="mask", **arm_kw):
        self.seeds = list(seeds)
        self.prefix = prefix
        self.arms = Arms(**arm_kw)
        inputs = self.arms.inputs
        self.p0 = hz.base_params()
        # Round-1 shape probabilities: the model scoring the public window (fold 0) is trained only
        # on other folds' pseudo-labels from the ball-only decode, which do not depend on the
        # window's ball. Round 2 leaks it (TECHNICAL.md section 6), so it is not used here.
        self.q = hz.frozen_shape(inputs, replace(self.p0, shape_rounds=1), "shape_round1")[:WINDOW]
        self.full_ball = inputs.atd["ball"].copy()
        self.arms.graded = inputs.graded.copy()
        windowed(inputs)
        self.inputs = inputs

    def prepare(self, force=False):
        for seed in self.seeds:
            names = ["{}_atd_s{}".format(self.prefix, seed), "{}_fused_s{}".format(self.prefix, seed)]
            if not force and all(os.path.isfile(os.path.join(hz.ROOT, "cache", "cv", n + ".npz"))
                                 for n in names):
                continue
            if seed < 0:
                ball, hidden = self.full_ball.copy(), np.zeros(len(self.full_ball), bool)
            else:
                ball, hidden, _ = transplant(self.full_ball, self.arms.graded,
                                             np.random.default_rng(100 + seed))
            ball, hidden = ball[:WINDOW], hidden[:WINDOW]
            fu = self.arms.fused(ball, hidden)
            xy = ball[:, :2].copy()
            pub = slice(*PUBLIC)
            xy[pub] = fu["xy"][pub]
            write_ball(names[0], ball[:, :2])
            write_ball(names[1], xy)

    def run(self, p, arm="fused"):
        out = []
        for seed in self.seeds:
            self.inputs._cache = {k: v for k, v in self.inputs._cache.items()
                                  if k[0] not in ("trk", "ev", "ownership")}
            pp = replace(p, ball_source="{}_{}_s{}".format(self.prefix, arm, seed))
            r = hz.score(hz.decode(self.inputs, pp, self.q, blocks=PUB))
            out.append((r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"]))
        return np.array(out)


def summary(tag, res):
    clean, masked = res[0], res[1:]
    return "{:34s} clean {:.4f}/{:.4f} n{:2d} | masked spell {:.4f} macro {:.4f} n {:.1f} | {}".format(
        tag, clean[0], clean[1], int(clean[2]), masked[:, 0].mean(), masked[:, 1].mean(),
        masked[:, 2].mean(), " ".join("{:.2f}".format(v) for v in masked[:, 0]))
