"""Fast experiment loop: the shape model's probabilities are computed once (with the submitted
params) and frozen; everything after it -- evidence, events, emissions, bonuses, decode -- is
recomputed per variant. Good for decoder and event changes; the shape model's self-training is
not re-run, so any interaction with it is ignored here (check finalists with src/run.py).
"""

import os
import sys
import time  # noqa: F401  (used by callers)
from dataclasses import replace  # noqa: F401

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
ROOT = os.path.dirname(SRC)
sys.path[:0] = [SRC, ROOT, os.path.join(ROOT, "evaluation")]

from possession.config import Params  # noqa: E402
from possession import pipeline, events, fuse, hsmm  # noqa: E402
import possession_metrics as pm  # noqa: E402

CACHE = os.path.join(ROOT, "cache", "lt")
BLOCKS = [(1, 11185, 90651), (2, 110808, 190197)]


def base_params(**kw):
    return Params.load(os.path.join(SRC, "params.json"), **kw)


def frozen_shape(inputs, p, name="shape_default"):
    path = os.path.join(CACHE, name + ".npy")
    if os.path.isfile(path):
        return np.load(path)
    os.makedirs(CACHE, exist_ok=True)
    _, dbg = pipeline.predict(inputs, p, debug=True)
    np.save(path, dbg["shape"])
    return dbg["shape"]


def decode(inputs, p, q, extra=None, blocks=None):
    """Labels from the final decode with frozen shape q. extra(emit, bonus, ctx) may edit both."""
    trk = inputs.tracking(p)
    ev = inputs.evidence(p)
    evs = events.build(trk, ev, inputs.tags, inputs.blocks, p)
    evs["tag_starts"] = inputs.tags["sp_start"]
    evs = fuse.add_whistles(evs, inputs.graded, p)
    bonus = fuse.bonuses(evs, inputs.n, p, ev)
    if p.supervised_shape_model:
        from possession import supervised_shape
        q = supervised_shape.replace_team(q, supervised_shape.infer(inputs, p))
    emit, p_dead = fuse.emissions(ev, q, inputs.tags, evs, p)
    emit = fuse.force_pre_kickoff(emit, inputs.game, p)
    if p.ownership_model:
        from possession import ownership
        emit, bonus = ownership.apply(emit, bonus, ownership.infer(inputs, p), p)
    if extra is not None:
        emit, bonus = extra(emit, bonus, {"trk": trk, "ev": ev, "events": evs, "p": p,
                                          "inputs": inputs, "p_dead": p_dead})
    model = hsmm.Model(p)
    labels, _ = model.decode(emit, bonus, blocks or inputs.blocks, inputs.n)
    return labels


_GT = None


def score(labels):
    global _GT
    if _GT is None:
        _GT = pipeline.public_gt()
    return pm.evaluate(_GT, pm.label_codes(labels))


def count(labels):
    n = 0
    for _, a, b in BLOCKS:
        x = labels[a:b + 1]
        ch = np.flatnonzero(x[1:] != x[:-1]) + 1
        starts = np.r_[0, ch]
        n += int(np.isin(x[starts], ["H", "A"]).sum())
    return n


def line(tag, labels):
    r = score(labels)
    return "{:28s} spell {:.4f}  macro {:.4f}  pub_poss {:2d}  full_poss {}".format(
        tag, r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"], count(labels))


def spell_table(variants):
    """Per GT possession of the public window: best IoU of a same-team predicted spell, per variant."""
    gt = pipeline.public_gt()
    s, e = 11185, 18685
    g = gt[s:e]

    def spells(x):
        out, st = [], 0
        for i in range(1, len(x) + 1):
            if i == len(x) or x[i] != x[st]:
                if x[st] in (0, 1):
                    out.append((x[st], st, i))
                st = i
        return out
    G = spells(g)
    P = {k: spells(pm.label_codes(v)[s:e]) for k, v in variants.items()}
    print("gt spell        dur  " + "  ".join("{:>8s}".format(k[:8]) for k in variants))
    for lab, a, b in G:
        row = []
        for k in variants:
            best = 0.0
            for pl, pa, pb in P[k]:
                if pl != lab:
                    continue
                inter = max(0, min(b, pb) - max(a, pa))
                best = max(best, inter / (max(b, pb) - min(a, pa)))
            row.append(best)
        print("{} {:6d}-{:6d} {:5.1f}  ".format("HA"[lab], a + s, b + s, (b - a) / 25)
              + "  ".join("{:8.2f}".format(v) for v in row))
