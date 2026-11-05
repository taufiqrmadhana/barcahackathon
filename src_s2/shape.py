"""Team-shape model: H / A / D probabilities from player positions only (no ball).

A team in possession pushes up and spreads; the other retreats and compacts; at a stoppage the
players walk. None of this needs the ball, so a classifier trained on frames where the ball IS
seen (labels = S2's own ball-based output, never ground truth) can fill the frames where it is
not. Cross-fitted over contiguous time blocks with a guard band: no frame is scored by a model
trained on it or on its neighbourhood. Idea and feature families follow Taufiq's
src/possession/shape.py (commit 875fe33); this is a smaller, self-contained version.
"""
import warnings

import numpy as np
from scipy.ndimage import uniform_filter1d
from sklearn.ensemble import HistGradientBoostingClassifier

from data import FPS, HALVES


def attack_sign(inp):
    """+1 where home attacks towards +x, -1 otherwise (per half, from mean team x)."""
    n = inp["n"]
    px, teams = inp["px"], inp["teams"]
    sign = np.ones(n)
    bounds = [0, (HALVES[0][1] + HALVES[1][0]) // 2, n]
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            xh = np.nanmean(px[lo:hi][:, teams == 0])
            xa = np.nanmean(px[lo:hi][:, teams == 1])
        sign[lo:hi] = 1.0 if xh < xa else -1.0  # home defends the low-x goal -> attacks +x
    return sign


def features(inp):
    px, py, teams = inp["px"], inp["py"], inp["teams"]
    L, W = inp["L"], inp["W"]
    sgn = attack_sign(inp)
    vx = np.full(px.shape, np.nan)
    vy = np.full(py.shape, np.nan)
    vx[2:-2] = (px[4:] - px[:-4]) / 4 * FPS
    vy[2:-2] = (py[4:] - py[:-4]) / 4 * FPS
    cols, names = [], []
    with warnings.catch_warnings(), np.errstate(invalid="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        per = []
        for side in (0, 1):
            s = sgn if side == 0 else -sgn
            m = teams == side
            x, y = px[:, m], py[:, m]
            u = np.where(s[:, None] > 0, x, L - x)
            srt = np.sort(u, 1)
            f = {
                "u_mean": np.nanmean(u, 1), "u_max": np.nanmax(u, 1), "u_def": srt[:, 1],
                "u_std": np.nanstd(u, 1), "y_std": np.nanstd(y, 1),
                "vu": np.nanmean(vx[:, m] * s[:, None], 1),
                "speed": np.nanmean(np.hypot(vx[:, m], vy[:, m]), 1),
                "outside": np.nansum((x < -1) | (x > L + 1) | (y < -1) | (y > W + 1), 1).astype(float),
                "half_opp": np.nansum(u > L / 2, 1).astype(float),
            }
            per.append(f)
        for tag, f in zip("ha", per):
            for k, v in f.items():
                cols.append(v)
                names.append(tag + "_" + k)
        for k in per[0]:
            cols.append(per[0][k] - per[1][k])
            names.append("d_" + k)
        # pressure: mean distance of each team's players to their nearest opponent
        h = np.stack([px[:, teams == 0], py[:, teams == 0]], -1)
        a = np.stack([px[:, teams == 1], py[:, teams == 1]], -1)
        hn = np.empty(len(px))
        an = np.empty(len(px))
        for lo in range(0, len(px), 20000):
            hi = min(len(px), lo + 20000)
            d = np.linalg.norm(h[lo:hi, :, None, :] - a[lo:hi, None, :, :], axis=-1)
            d = np.where(np.isnan(d), np.inf, d)
            hmin, amin = d.min(2), d.min(1)
            hmin[np.isinf(hmin)] = np.nan
            amin[np.isinf(amin)] = np.nan
            hn[lo:hi] = np.nanmean(hmin, 1)
            an[lo:hi] = np.nanmean(amin, 1)
        cols += [hn, an]
        names += ["h_press", "a_press"]
    base = np.stack(cols, 1)
    base = np.where(np.isnan(base), np.nanmean(base, 0), base)
    out, onames = [base], list(names)
    for w in (25, 75):
        sm = uniform_filter1d(base, w, axis=0)
        out.append(sm)
        onames += [k + "_s%d" % w for k in names]
        lag = np.vstack([np.repeat(sm[:1], w, 0), sm[:-w]])
        out.append(sm - lag)
        onames += [k + "_d%d" % w for k in names]
    return np.hstack(out).astype(np.float32), onames


def fit_predict(X, y, train, folds=10, guard=750, stride=4, iters=150):
    """Cross-fitted (N, 3) probabilities for classes 0/1/2 (H/A/D) over the graded frames."""
    n = len(X)
    graded = np.zeros(n, bool)
    for a, b in HALVES:
        graded[a:b] = True
    idx = np.flatnonzero(graded)
    edges = np.linspace(0, len(idx), folds + 1).astype(int)
    proba = np.full((n, 3), 1 / 3, np.float32)
    for k in range(folds):
        blk = idx[edges[k]:edges[k + 1]]
        lo, hi = blk[0], blk[-1] + 1
        tr = train.copy()
        tr[max(0, lo - guard):hi + guard] = False
        tr = np.flatnonzero(tr)[::stride]
        mdl = HistGradientBoostingClassifier(max_iter=iters, learning_rate=0.08, max_leaf_nodes=31,
                                             l2_regularization=1.0, early_stopping=False,
                                             random_state=0)
        mdl.fit(X[tr], y[tr])
        pr = mdl.predict_proba(X[lo:hi])
        out = np.zeros((hi - lo, 3), np.float32)
        for j, c in enumerate(mdl.classes_):
            out[:, c] = pr[:, j]
        proba[lo:hi] = out
    return proba


VERSION = 1


def get_proba(inp, p):
    """Shape probabilities for the whole match, cross-fitted; pseudo-labels = S2 without the shape
    model, on frames where the ball is seen and >= 12 frames from a label change. Cached."""
    import hashlib
    import json
    import os

    import s2
    base = {k: v for k, v in p.items() if not k.startswith("shape") and k != "restart_win"}
    # Key on the source too: a params-only key served stale probabilities after code changes.
    here = os.path.dirname(os.path.abspath(__file__))
    source = b"".join(open(os.path.join(here, f), "rb").read() for f in ("s2.py", "shape.py"))
    key = hashlib.md5((json.dumps(base, sort_keys=True) + str(VERSION)).encode() + source).hexdigest()[:10]
    cache = os.path.join(os.environ.get("BARCA_CACHE_DIR", "."), "s2shape_{}.npy".format(key))
    if os.path.isfile(cache):
        return np.load(cache)
    q = dict(inp)
    q["shape"] = None
    lab = s2.predict(q, s2.Params(base))
    X, _ = features(inp)
    seen = ~np.isnan(inp["bx"])
    edge = np.zeros(len(lab), bool)
    for c in np.flatnonzero(np.diff(lab)) + 1:
        edge[max(0, c - 12):c + 12] = True
    pr = fit_predict(X, lab, seen & ~edge)
    np.save(cache, pr)
    return pr
