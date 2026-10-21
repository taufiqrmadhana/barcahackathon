"""Data in, labels out. Shared by the CLI, the tuner and the analysis script.

    tracking -> clean -> evidence -> events ---------------------------> bonuses --+
                                  \\-> emissions (ball only) -> decode #1          |
                                                                  |              v
                              shape model trained on decode #1 ---+-> emissions -> decode #2
"""

import hashlib
import os
from dataclasses import asdict

import numpy as np
import pandas as pd

from . import atd as atd_io
from . import clean, evidence, events, fuse, hsmm, shape, tagging
from .paths import CACHE_DIR, ROOT
from games import GAME, half_video_blocks, source_frames
import possession_metrics as pm

PUBLIC_GT = os.path.join(ROOT, "evaluation", "gt", "{}_possession_gt_public.csv".format(GAME))
CODES = {"H": 0, "A": 1, "D": 2}

CLEAN_KEYS = ("player_margin", "max_player_speed", "vel_k", "ball_spike_speed",
              "ball_island_len", "ball_island_gap", "max_gap")
EVIDENCE_KEYS = ("ctrl_radius", "ctrl_softness", "sync_free", "sync_scale", "filled_trust",
                 "corner_zone", "still_speed", "slow_window")


def _key(params, names):
    values = asdict(params)
    return tuple(values[k] for k in names)


class Inputs:
    """Loaded data plus intermediate results cached by the params they depend on."""

    def __init__(self, game=GAME, data_dir=None, refresh=False):
        self.game = game
        self.atd = atd_io.load(game, data_dir, refresh)
        self.tags = tagging.priors(game, data_dir)
        self.blocks = half_video_blocks(game)
        self.n = source_frames("tactical", game)
        self.graded = np.zeros(self.n, dtype=bool)
        for _, first, last in self.blocks:
            self.graded[first:last + 1] = True
        self._cache = {}

    def _get(self, name, key, build):
        full = (name,) + key
        if full not in self._cache:
            self._cache[full] = build()
        return self._cache[full]

    def tracking(self, p):
        return self._get("trk", _key(p, CLEAN_KEYS),
                         lambda: clean.build(self.atd, self.blocks, p))

    def evidence(self, p):
        key = _key(p, CLEAN_KEYS + EVIDENCE_KEYS)
        return self._get("ev", key, lambda: evidence.build(self.tracking(p), p))

    def shape_features(self, p):
        key = _key(p, CLEAN_KEYS + EVIDENCE_KEYS)
        return self._get("shape_x", key,
                         lambda: shape.features(self.tracking(p), self.evidence(p)))

    def shape_proba(self, p, labels, train):
        codes = np.array([CODES.get(c, -1) for c in labels], dtype=np.int8)
        X, names = self.shape_features(p)
        digest = hashlib.sha1(codes.tobytes() + train.tobytes() + repr(
            _key(p, CLEAN_KEYS + EVIDENCE_KEYS + ("shape_folds", "shape_guard", "shape_stride",
                                                  "shape_iter"))).encode()
            + ",".join(names).encode()).hexdigest()[:16]
        path = os.path.join(CACHE_DIR, "shape_{}_{}.npy".format(self.game, digest))
        if os.path.isfile(path):
            return np.load(path)
        q = shape.fit_predict(X, codes, train & (codes >= 0), self.graded, p)
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.save(path, q)
        return q


def shape_training_mask(inputs, ev, labels, p):
    """Frames whose decode-#1 label comes from a tracked ball and is not next to a change."""
    mask = inputs.graded & ev["seen"]
    change = np.zeros(len(labels), dtype=bool)
    change[1:] = labels[1:] != labels[:-1]
    near = np.convolve(change.astype(np.int8), np.ones(51, dtype=np.int8), mode="same") > 0
    return mask & ~near


def predict(inputs, p, debug=False):
    trk = inputs.tracking(p)
    ev = inputs.evidence(p)
    evs = events.build(trk, ev, inputs.tags, inputs.blocks, p)
    evs["tag_starts"] = inputs.tags["sp_start"]
    model = hsmm.Model(p)
    bonus = fuse.bonuses(evs, inputs.n, p)

    emit1, _ = fuse.emissions(ev, None, inputs.tags, evs, p)
    labels1, _ = model.decode(emit1, bonus, inputs.blocks, inputs.n)

    labels = labels1
    for _ in range(max(1, p.shape_rounds)):
        train = shape_training_mask(inputs, ev, labels, p)
        q = inputs.shape_proba(p, labels, train)
        emit2, p_dead = fuse.emissions(ev, q, inputs.tags, evs, p)
        labels, states = model.decode(emit2, bonus, inputs.blocks, inputs.n)
    if not debug:
        return labels
    return labels, {"trk": trk, "ev": ev, "events": evs, "bonus": bonus, "emit": emit2,
                    "p_dead": p_dead, "shape": q, "labels1": labels1, "states": states,
                    "train": train}


def write(path, labels):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    table = pd.DataFrame({"frame": np.arange(len(labels)), "label": labels})
    problem = pm.validate_submission(table, source_frames("tactical"))
    if problem:
        raise ValueError("invalid submission: " + problem)
    table.to_csv(path, index=False, lineterminator="\n")
    return path


def public_gt():
    return pm.label_codes(pm.read_label_csv(PUBLIC_GT))


def score(labels, gt_codes=None):
    gt_codes = public_gt() if gt_codes is None else gt_codes
    return pm.evaluate(gt_codes, pm.label_codes(labels))
