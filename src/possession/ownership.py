"""Supervised ball-centred team evidence, independent of the self-trained shape model.

Each team is evaluated through the same binary classifiers (own team versus opponent).
Features contain neither player identities nor absolute team names. D is not a target:
ownership is conditional on alive play; boundaries are direct H<->A changes only.
The first experiment adds evidence to the frozen baseline, rather than simultaneously
retraining its shape model. It runs only in the final decode.
"""

import hashlib
from dataclasses import asdict

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from .clean import FPS, since_last, until_next, velocity

VERSION = 1
OFFSETS = (-25, -12, 0, 12, 25)
CLEAN_KEYS = ("player_margin", "max_player_speed", "vel_k", "ball_spike_speed",
              "ball_island_len", "ball_island_gap", "max_gap")


def features(trk, blocks, k=3):
    """Return (two team views, support, names), using +/-1 s without crossing halves.

    Distances, speed magnitudes and radial motion are invariant to pitch reflections.
    Cleaned velocity stencils near half edges are excluded from these features.
    Missing values stay NaN for the GBM; there is no whole-game imputation statistic.
    """
    ball = trk["ball"]
    xy, teams = trk["players_xy"], trk["teams"]
    bxy = ball["xy"]
    n = len(bxy)
    bv, pv = np.full_like(bxy, np.nan), np.full_like(xy, np.nan)
    age, ahead = np.full(n, 60.0), np.full(n, 60.0)
    accel = np.full(n, np.nan)
    for _, first, last in blocks:
        sl = slice(first, last + 1)
        # Use the pipeline's cleaned velocities, excluding its cross-half stencil.
        bv[sl], pv[sl] = ball["vel"][sl], trk["players_vel"][sl]
        # The cleaner's stencil may be configured; invalidate 1 s at each edge.
        # Context itself is padded with NaN below, not copied from another half.
        for a, b in ((first, min(last + 1, first + 25)), (max(first, last - 24), last + 1)):
            bv[a:b], pv[a:b] = np.nan, np.nan
        age[sl] = np.minimum(since_last(ball["seen"][sl]) / FPS, 60)
        ahead[sl] = np.minimum(until_next(ball["seen"][sl]) / FPS, 60)
        av = velocity(bv[sl], 3)
        accel[sl] = np.linalg.norm(av, axis=1)

    length, width = trk["pitch"]
    line_x = np.minimum(bxy[:, 0], length - bxy[:, 0])
    line_y = np.minimum(bxy[:, 1], width - bxy[:, 1])
    common = np.column_stack([ball["known"], ball["seen"], age, ahead,
                              np.linalg.norm(bv, axis=1), accel, line_x, line_y])
    common_names = ["known", "seen", "since_seen_s", "until_seen_s", "ball_speed",
                    "ball_accel", "goal_line_distance", "touchline_distance"]
    sides = []
    for side in (0, 1):
        cols = np.flatnonzero(teams == side)
        if len(cols) < k:
            raise ValueError("ownership features need at least three slots per team")
        delta = xy[:, cols] - bxy[:, None]
        distance = np.linalg.norm(delta, axis=2)
        order = np.argsort(np.where(np.isfinite(distance), distance, np.inf), axis=1)[:, :k]
        rows = np.arange(n)[:, None]
        dist = distance[rows, order]
        pos = delta[rows, order]
        vel = pv[:, cols][rows, order]
        rel = vel - bv[:, None]
        radial = np.sum(rel * pos, axis=2) / np.maximum(dist, 0.1)
        alignment = np.sum(vel * bv[:, None], axis=2) / np.maximum(
            np.linalg.norm(vel, axis=2) * np.linalg.norm(bv, axis=1)[:, None], 0.1)
        items = np.stack([dist, np.linalg.norm(rel, axis=2), np.linalg.norm(vel, axis=2),
                          radial, alignment], axis=2)
        items[~np.isfinite(dist)] = np.nan
        sides.append(items.reshape(n, -1))
    near_names = [f"nearest_{j}_{name}" for j in range(k)
                  for name in ("dist", "relative_speed", "speed", "radial_speed", "alignment")]
    base_names = common_names + [f"own_{s}" for s in near_names] + [f"opp_{s}" for s in near_names]
    views = []
    for side in (0, 1):
        base = np.column_stack([common, sides[side], sides[1 - side]]).astype(np.float32)
        context = []
        for offset in OFFSETS:
            shifted = np.full_like(base, np.nan)
            for _, first, last in blocks:
                lo, hi = max(first, first - offset), min(last + 1, last + 1 - offset)
                if hi > lo:
                    shifted[lo:hi] = base[lo + offset:hi + offset]
            context.append(shifted)
        views.append(np.concatenate(context, axis=1))
    # A ball-centred model has no evidence deep inside an occlusion. Leave shape in charge.
    support = (age <= 1.0) | (ahead <= 1.0)
    names = [f"{name}@{offset}" for offset in OFFSETS for name in base_names]
    return views, support, names


def targets(labels, blocks, radius=10):
    """Directed boundary neighbourhoods (+/-0.4 s); never label D/X as an owner.

    Transition training uses alive frames, including negatives around D; only direct
    H/A changes make positive labels. Overlapping H/A neighbourhoods may both be positive.
    """
    owner = np.full(len(labels), -1, dtype=np.int8)
    owner[labels == "H"], owner[labels == "A"] = 0, 1
    boundary = np.zeros((len(labels), 2), dtype=np.int8)
    valid = np.zeros(len(labels), dtype=bool)
    for _, first, last in blocks:
        valid[first:last + 1] = True
        change = np.flatnonzero((owner[first + 1:last + 1] >= 0)
                               & (owner[first:last] >= 0)
                               & (owner[first + 1:last + 1] != owner[first:last])) + first + 1
        for t in change:
            boundary[max(first, t - radius):min(last + 1, t + radius + 1), owner[t]] = 1
    return owner, boundary, valid & (owner >= 0)


def fit(batches, params, train_games, seed=20261026):
    """batches: (views, support, names, labels, blocks); no held-out game enters here."""
    xs, ys, ts = [], [], []
    names = batches[0][2]
    for views, support, feature_names, labels, blocks in batches:
        if feature_names != names:
            raise ValueError("feature schema mismatch")
        own, boundary, alive = targets(labels, blocks)
        # Deterministic 5 Hz samples. No frame-random validation or early stopping.
        idx = np.flatnonzero(alive & support)[::5]
        for side in (0, 1):
            xs.append(views[side][idx])
            ys.append((own[idx] == side).astype(np.int8))
            ts.append(boundary[idx, side])
    x, y, t = np.concatenate(xs), np.concatenate(ys), np.concatenate(ts)
    models = []
    for target in (y, t):
        if len(np.unique(target)) != 2:
            raise ValueError("training needs positive and negative owner/boundary examples")
        model = HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=15,
                    min_samples_leaf=60, learning_rate=0.08, l2_regularization=5.0,
                    early_stopping=False, random_state=seed)
        model.fit(x, target)
        models.append(model)
    return {"version": VERSION, "names": names, "models": models,
            "cleaning": {key: asdict(params)[key] for key in CLEAN_KEYS},
            "train_games": sorted(train_games), "seed": seed,
            "training_rows": len(y), "boundary_positive_share": float(t.mean())}


def proba(artifact, batch):
    views, support, names = batch
    if artifact["version"] != VERSION or artifact["names"] != names:
        raise ValueError("ownership checkpoint feature schema mismatch")
    owner, transition = [np.column_stack([model.predict_proba(x)[:, 1] for x in views])
                         for model in artifact["models"]]
    owner /= np.maximum(owner.sum(axis=1, keepdims=True), 1e-9)
    owner[~support], transition[~support] = 0.5, 0.0
    return owner, transition, support


def infer(inputs, params):
    with open(params.ownership_model, "rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    cleaning = {key: asdict(params)[key] for key in CLEAN_KEYS}
    def build():
        artifact = joblib.load(params.ownership_model)
        if artifact["cleaning"] != cleaning:
            raise ValueError("ownership checkpoint cleaning differs from inference params")
        return proba(artifact, features(inputs.tracking(params), inputs.blocks))
    # The ball track the features are built from is part of the key (ball_source).
    return inputs._get("ownership", (digest, tuple(cleaning.values()), params.ball_source), build)


def apply(emit, bonus, predictions, params):
    """Add team evidence to BOTH ctrl/loose states; preserve D emissions and D bonuses.

    This is an additive first-stage ablation. The old ball-control evidence remains.
    Learned transition evidence competes with existing touch evidence via max, so the
    same event does not earn two switch bonuses. Probabilities denote a neighbourhood,
    not a calibrated per-frame switch hazard.
    """
    owner, transition, support = predictions
    emit, bonus = emit.copy(), bonus.copy()
    score = np.log(np.clip(owner, 0.02, 0.98)) - np.log(0.5)
    score[~support] = 0
    emit[:, :2] += params.w_owner * score[:, 0, None]
    emit[:, 2:4] += params.w_owner * score[:, 1, None]
    bonus[:, :2] = np.maximum(bonus[:, :2], params.w_owner_switch * transition)
    return emit, bonus
