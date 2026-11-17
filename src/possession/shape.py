"""Who has the ball when the ball is not tracked: read it off the two teams' shapes.

A team in possession pushes forward and spreads out, the team without it retreats and compacts.
None of that needs the ball, so a classifier trained on frames where the ball is tracked (labels
from the ball-based decode, never from the ground truth) can be applied to the frames where it
is not. Predictions are cross-fitted over time blocks: no frame is scored by a model that saw it.
"""

import warnings

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from .evidence import smooth

HOME, AWAY = 0, 1


def _team_features(xy, vel, teams, attack, length, side):
    sign = attack.astype(np.float32) * (1.0 if side == HOME else -1.0)
    cols = teams == side
    x, y = xy[:, cols, 0], xy[:, cols, 1]
    u = np.where(sign[:, None] > 0, x, length - x)
    vu = vel[:, cols, 0] * sign[:, None]
    speed = np.hypot(vel[:, cols, 0], vel[:, cols, 1])
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        srt = np.sort(u, axis=1)
        count = (~np.isnan(u)).sum(axis=1)
        feats = {
            "u_mean": np.nanmean(u, axis=1),
            "u_max": np.nanmax(u, axis=1),
            "u_min": np.nanmin(u, axis=1),
            "u_def": srt[:, 1],
            "u_std": np.nanstd(u, axis=1),
            "y_std": np.nanstd(y, axis=1),
            "vu": np.nanmean(vu, axis=1),
            "speed": np.nanmean(speed, axis=1),
            "half_opp": np.nansum(u > length / 2.0, axis=1).astype(np.float32),
            "count": count.astype(np.float32),
        }
    return feats


def _pressure(xy, teams, chunk=20000):
    """Per team: mean and min distance from its players to the nearest opponent."""
    n = len(xy)
    out = np.full((n, 4), np.nan, dtype=np.float32)
    home, away = xy[:, teams == HOME], xy[:, teams == AWAY]
    for lo in range(0, n, chunk):
        hi = min(n, lo + chunk)
        d = np.hypot(*(home[lo:hi, :, None, :] - away[lo:hi, None, :, :]).transpose(3, 0, 1, 2))
        d = np.where(np.isnan(d), np.inf, d)
        h_near, a_near = d.min(axis=2), d.min(axis=1)
        h_near[np.isinf(h_near)] = np.nan
        a_near[np.isinf(a_near)] = np.nan
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out[lo:hi] = np.stack([np.nanmean(h_near, 1), np.nanmin(h_near, 1),
                                   np.nanmean(a_near, 1), np.nanmin(a_near, 1)], axis=1)
    return out


def features(trk, ev, windows=(25, 75)):
    length, _ = trk["pitch"]
    xy, vel, teams, attack = trk["players_xy"], trk["players_vel"], trk["teams"], trk["attack"]
    cols, names = [], []
    per_team = [_team_features(xy, vel, teams, attack, length, s) for s in (HOME, AWAY)]
    for side, feats in zip(("h", "a"), per_team):
        for key, value in feats.items():
            cols.append(value)
            names.append(side + "_" + key)
    for key in per_team[0]:
        cols.append(per_team[0][key] - per_team[1][key])
        names.append("d_" + key)
    press = _pressure(xy, teams)
    for j, key in enumerate(("h_press_mean", "h_press_min", "a_press_mean", "a_press_min")):
        cols.append(press[:, j])
        names.append(key)
    cols.append(ev["player_speed_smooth"])
    names.append("speed_smooth")

    base = np.stack(cols, axis=1).astype(np.float32)
    out, out_names = [base], list(names)
    for w in windows:
        sm = np.stack([smooth(np.nan_to_num(c, nan=np.nanmean(c)), w) for c in base.T], axis=1)
        out.append(sm.astype(np.float32))
        out_names += ["{}_s{}".format(nm, w) for nm in names]
        lag = np.vstack([np.repeat(sm[:1], w, axis=0), sm[:-w]])
        out.append((sm - lag).astype(np.float32))
        out_names += ["{}_d{}".format(nm, w) for nm in names]
    return np.hstack(out), out_names


def fit_predict(X, labels, train_mask, graded, p):
    """(N, 3) probabilities of H / A / D, cross-fitted over contiguous time blocks."""
    proba = np.full((len(X), 3), np.nan, dtype=np.float32)
    idx = np.flatnonzero(graded)
    edges = np.linspace(0, len(idx), p.shape_folds + 1).astype(int)
    stride = p.shape_stride
    for k in range(p.shape_folds):
        block = idx[edges[k]:edges[k + 1]]
        lo, hi = block[0], block[-1] + 1
        guard = p.shape_guard
        train = train_mask.copy()
        train[max(0, lo - guard):hi + guard] = False
        tr = np.flatnonzero(train)[::stride]
        if len(np.unique(labels[tr])) < 2:
            continue
        model = HistGradientBoostingClassifier(
            max_iter=p.shape_iter, learning_rate=0.08, max_leaf_nodes=31,
            l2_regularization=1.0, early_stopping=False, random_state=0)
        model.fit(X[tr], labels[tr])
        pr = model.predict_proba(X[lo:hi])
        for j, c in enumerate(model.classes_):
            proba[lo:hi, c] = pr[:, j]
    proba = np.nan_to_num(proba, nan=0.0)
    total = proba.sum(axis=1, keepdims=True)
    proba = np.where(total > 0, proba / np.maximum(total, 1e-9), 1.0 / 3.0)
    return proba
