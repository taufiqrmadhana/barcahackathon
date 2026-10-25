"""Players-only conditional ownership, supervised on external event labels.

The D probability is deliberately supplied by the existing pipeline. This separate
ablation replaces only its H/A ratio, without feeding external predictions back into
pseudo-label fitting. Features are isolated by half and never read ball coordinates.
"""
from dataclasses import asdict
import hashlib

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from . import clean, evidence, shape

VERSION = 1
KEYS = ("player_margin", "max_player_speed", "vel_k", "slow_window")


def features(inputs, p):
    def build():
        chunks, names = [], None
        atd = inputs.atd
        for pid, first, last in inputs.blocks:
            players = atd["players"][first:last + 1]
            xy, vel = clean.clean_players(players, np.asarray(atd["pitch"]), atd["pitch"], p)
            block = [(pid, 0, len(players) - 1)]
            trk = {"players_xy": xy, "players_vel": vel, "teams": atd["teams"],
                   "pitch": atd["pitch"],
                   "attack": clean.attack_sign(xy, atd["teams"], block, atd["pitch"])}
            speed = evidence.team_motion(xy, vel, atd["teams"])["speed"]
            x, names = shape.features(trk, {"player_speed_smooth": evidence.smooth(speed, p.slow_window)})
            chunks.append((first, last, x))
        all_x = np.full((inputs.n, len(names)), np.nan, dtype=np.float32)
        for first, last, x in chunks:
            all_x[first:last + 1] = x
        return all_x, names
    return inputs._get("external_shape_features", tuple(asdict(p)[key] for key in KEYS), build)


def swap_view(x, names):
    """Exchange team-relative feature groups, negate home-minus-away differences."""
    index = {name: i for i, name in enumerate(names)}
    order, signs = [], []
    for name in names:
        target = "a_" + name[2:] if name.startswith("h_") else (
                 "h_" + name[2:] if name.startswith("a_") else name)
        order.append(index[target])
        signs.append(-1 if name.startswith("d_") else 1)
    return x[:, order] * np.asarray(signs, dtype=np.float32)


def fit(batches, p, train_games, seed=20261026):
    xs, ys = [], []
    names = batches[0][1]
    for x, feature_names, labels in batches:
        if feature_names != names:
            raise ValueError("shape feature schema differs between matches")
        idx = np.flatnonzero(np.isin(labels, ["H", "A"]))[::5]
        y = (labels[idx] == "H").astype(np.int8)
        xs.extend([x[idx], swap_view(x[idx], names)])
        ys.extend([y, 1 - y])
    x, y = np.concatenate(xs), np.concatenate(ys)
    model = HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=15,
                min_samples_leaf=120, learning_rate=0.08, l2_regularization=5.0,
                early_stopping=False, random_state=seed)
    model.fit(x, y)
    return {"version": VERSION, "model": model, "names": names,
            "params": {key: asdict(p)[key] for key in KEYS},
            "train_games": sorted(train_games), "seed": seed, "training_rows": len(y)}


def proba(artifact, features):
    x, names = features
    if artifact["version"] != VERSION or artifact["names"] != names:
        raise ValueError("supervised shape checkpoint feature schema mismatch")
    model = artifact["model"]
    home = model.predict_proba(x)[:, 1]
    away = model.predict_proba(swap_view(x, names))[:, 1]
    return np.clip(home / np.maximum(home + away, 1e-9), 0.02, 0.98)


def infer(inputs, p):
    with open(p.supervised_shape_model, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    params = {key: asdict(p)[key] for key in KEYS}
    def build():
        artifact = joblib.load(p.supervised_shape_model)
        if artifact["params"] != params:
            raise ValueError("shape checkpoint preprocessing parameters differ")
        return proba(artifact, features(inputs, p))
    return inputs._get("external_shape_proba", (digest, tuple(params.values())), build)


def replace_team(q, home):
    out = q.copy()
    alive = q[:, 0] + q[:, 1]
    out[:, 0], out[:, 1] = alive * home, alive * (1 - home)
    return out
