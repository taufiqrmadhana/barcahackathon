"""How the pipeline holds up when the ball disappears the way it does in the rest of the match.

The public window has the ball 88% of the time, the full match 59%. This hides the ball on extra
gaps inside the public window until a target share is missing, then reruns the WHOLE pipeline on
the masked tracking: cleaning, evidence, events, pseudo-labels, every shape-model fit and every
self-training round. Nothing downstream ever sees the hidden ball. (--frozen reproduces the old,
leaky shortcut that reused shape probabilities from the unmasked run, to measure the difference.)

Gap models:
  uniform    log-uniform lengths 3-40 s, placed anywhere
  empirical  lengths resampled from the real gaps outside the public window, placed anywhere
  context    empirical lengths, placed where the real tracking tends to lose the ball: a hazard
             model fitted outside the public window on what the ball was doing 8 frames before
             each real gap (speed, nearest players, duel, distance to the lines, crowding)
  transplant the real missing-ball mask of another five minutes of the match, laid over the
             window as it is: real lengths, real clustering, but not the real situations

This measures robustness to simulated gaps. It is not an estimate of the private score: real
gaps may still sit in harder moments than any of these models capture.

    python src/robustness.py --sim empirical context --seeds 3
"""

import argparse
import copy
import json
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from possession.config import Params  # noqa: E402
from possession.clean import runs  # noqa: E402
from possession import pipeline  # noqa: E402
from run import DEFAULT_PARAMS, parse_overrides  # noqa: E402

PUBLIC = (11185, 18685)
MIN_GAP = 75
LEAD = 8
HAZARD_FEATURES = ("ball_speed", "nearest", "both_close", "line", "crowd", "player_speed")


def real_gaps(inputs):
    """Lengths of the real missing-ball gaps (>= MIN_GAP) in graded frames outside the window."""
    missing = np.isnan(inputs.atd["ball"][:, 0]) & inputs.graded
    out = []
    for start, stop in zip(*runs(missing)):
        if stop - start >= MIN_GAP and (stop <= PUBLIC[0] or start >= PUBLIC[1]):
            out.append(stop - start)
    return np.array(out)


def hazard_table(inputs, params):
    """(features, gap-starts-next flag) for every frame with the ball, from the real tracking."""
    ev = inputs.evidence(params)
    trk = inputs.tracking(params)
    seen = np.isnan(inputs.atd["ball"][:, 0]) == False  # noqa: E712
    length, width = trk["pitch"]
    xy = trk["ball"]["xy"]
    players = trk["players_xy"]
    crowd = np.zeros(len(xy), dtype=np.float32)
    rows = np.flatnonzero(seen)
    for lo in range(0, len(rows), 20000):
        r = rows[lo:lo + 20000]
        d = np.hypot(*(players[r] - xy[r, None, :]).transpose(2, 0, 1))
        crowd[r] = (np.nan_to_num(d, nan=99.0) < 5.0).sum(axis=1)
    x = np.stack([
        ev["ball_speed_smooth"],
        np.minimum(ev["dist"].min(axis=1), 30.0),
        (ev["dist"].max(axis=1) < 2.0).astype(np.float32),
        np.clip(np.minimum.reduce([xy[:, 0], length - xy[:, 0], xy[:, 1], width - xy[:, 1]]),
                -5.0, 35.0),
        crowd,
        ev["player_speed_smooth"],
    ], axis=1)
    x = np.nan_to_num(x, nan=0.0)
    x = np.vstack([np.repeat(x[:LEAD], LEAD, axis=0)[:LEAD], x[:-LEAD]])
    start = np.zeros(len(xy), dtype=bool)
    missing = ~seen
    for a, b in zip(*runs(missing)):
        if b - a >= MIN_GAP and a > 0:
            start[a - 1] = True
    return x, start, seen


def fit_hazard(inputs, params):
    x, start, seen = hazard_table(inputs, params)
    outside = inputs.graded & seen
    outside[PUBLIC[0] - 250:PUBLIC[1] + 250] = False
    scaler = StandardScaler().fit(x[outside])
    model = LogisticRegression(C=1.0, max_iter=1000).fit(scaler.transform(x[outside]),
                                                         start[outside])
    weights = np.zeros(len(x))
    weights[seen] = model.predict_proba(scaler.transform(x[seen]))[:, 1]
    odds = dict(zip(HAZARD_FEATURES, np.exp(model.coef_[0]).round(2).tolist()))
    return weights, odds, int(start[outside].sum())


def hide_ball(atd, rng, share, lengths=None, weights=None, lo=PUBLIC[0], hi=PUBLIC[1]):
    """Copy of the ATD with extra ball gaps in [lo, hi) until `share` of it is missing."""
    ball = atd["ball"].copy()
    hidden = np.zeros(len(ball), dtype=bool)
    for _ in range(10000):
        if np.isnan(ball[lo:hi, 0]).mean() >= share:
            break
        if lengths is None:
            length = int(np.exp(rng.uniform(np.log(MIN_GAP), np.log(1000))))
        else:
            length = int(rng.choice(lengths))
        length = min(length, hi - lo - 1)
        if weights is None:
            start = int(rng.integers(lo, hi - length))
        else:
            w = weights[lo:hi - length] * ~np.isnan(ball[lo:hi - length, 0])
            if w.sum() <= 0:
                break
            start = lo + int(rng.choice(len(w), p=w / w.sum())) + 1
        hidden[start:start + length] |= ~np.isnan(ball[start:start + length, 0])
        ball[start:start + length] = np.nan
    out = dict(atd)
    out["ball"] = ball
    return out, hidden


def transplant(inputs, rng, lo=PUBLIC[0], hi=PUBLIC[1]):
    """Copy of the ATD with the real missing-ball mask of another graded stretch laid over."""
    n = hi - lo
    graded = inputs.graded.copy()
    graded[lo - n:hi] = False
    starts = [s for s in np.flatnonzero(graded) if graded[s:s + n].all()]
    src = int(rng.choice(starts[::25]))
    mask = np.isnan(inputs.atd["ball"][src:src + n, 0])
    ball = inputs.atd["ball"].copy()
    hidden = np.zeros(len(ball), dtype=bool)
    hidden[lo:hi] = mask & ~np.isnan(ball[lo:hi, 0])
    ball[lo:hi][mask] = np.nan
    out = dict(inputs.atd)
    out["ball"] = ball
    return out, hidden


def masked_inputs(inputs, atd):
    other = copy.copy(inputs)
    other.atd = atd
    other._cache = {}
    return other


def hidden_metrics(labels, gt, hidden):
    alive = hidden & np.isin(gt, ["H", "A"])
    dead = hidden & (gt == "D")
    pred_dead = hidden & (labels == "D")
    return {
        "hidden_frames": int(hidden.sum()),
        "hidden_gt_dead_share": float(dead.sum() / max(1, hidden.sum())),
        "hidden_team_acc": float((labels[alive] == gt[alive]).mean()) if alive.any() else None,
        "hidden_dead_recall": float((labels[dead] == "D").mean()) if dead.any() else None,
        "hidden_dead_precision": float((gt[pred_dead] == "D").mean()) if pred_dead.any()
        else None,
    }


def evaluate(inputs, params, gt_codes, gt, hidden=None, frozen_q=None):
    if frozen_q is None:
        labels, dbg = pipeline.predict(inputs, params, debug=True)
    else:
        from tune import decode_with
        labels, dbg = decode_with(inputs, params, frozen_q), None
    r = pipeline.score(labels, gt_codes)
    out = {"spellF1": r["spell_f1_mean"], "macroF1": r["macro_f1"], "poss": r["n_pred_spells"]}
    if hidden is not None:
        out.update(hidden_metrics(labels, gt, hidden))
    if dbg is not None:
        import spells
        rows, extra = spells.report(inputs, params, gt_codes, verbose=False, result=(labels, dbg))
        cats = {}
        for row in rows:
            key = row["cat"] if row["cat"] != "missed" else "missed:" + row["why"].split(" (")[0]
            cats[key] = cats.get(key, 0) + 1
        out["spells"] = cats
        out["spurious"] = extra
        out["short_hit"] = sum(r["cat"] == "hit" for r in rows if r["dur"] < 4)
        graded = labels[inputs.graded]
        starts = np.r_[0, np.flatnonzero(graded[1:] != graded[:-1]) + 1]
        out["full_poss"] = int((graded[starts] != "D").sum())
    return out, labels


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--params", default=DEFAULT_PARAMS)
    parser.add_argument("--set", nargs="+", metavar="KEY=VALUE")
    parser.add_argument("--missing", type=float, default=0.41)
    parser.add_argument("--sim", nargs="+", default=["empirical", "context"],
                        choices=["uniform", "empirical", "context", "transplant"])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--frozen", action="store_true",
                        help="old shortcut: reuse shape probabilities from the unmasked run")
    parser.add_argument("--json", help="append one line per run to this file")
    args = parser.parse_args()

    params = Params.load(args.params if os.path.isfile(args.params) else None,
                         **parse_overrides(args.set))
    inputs = pipeline.Inputs()
    gt_codes = pipeline.public_gt()
    gt = np.array(["H", "A", "D", "X"])[gt_codes]

    clean, clean_labels = evaluate(inputs, params, gt_codes, gt)
    if args.json:
        with open(args.json, "a") as handle:
            handle.write(json.dumps(dict(clean, sim="none", seed=-1, frozen=False,
                                         set=args.set or [])) + "\n")
    frozen_q = None
    if args.frozen:
        frozen_q = pipeline.predict(inputs, params, debug=True)[1]["shape"]
    print("no ball hidden   spellF1 {:.4f}  macroF1 {:.4f}  poss {}  spells {}  full-match "
          "possessions {}".format(clean["spellF1"], clean["macroF1"], clean["poss"],
                                  clean.get("spells"), clean.get("full_poss")))

    lengths = real_gaps(inputs)
    natural = np.isnan(inputs.atd["ball"][PUBLIC[0]:PUBLIC[1], 0])
    print("real gaps outside the window: {}  length median {:.1f}s  p90 {:.1f}s  max {:.1f}s"
          .format(len(lengths), np.median(lengths) / 25, np.percentile(lengths, 90) / 25,
                  lengths.max() / 25))
    print("ball already missing in the window: {:.1%}  of it gt D: {:.1%}".format(
        natural.mean(), (gt[PUBLIC[0]:PUBLIC[1]][natural] == "D").mean()))
    weights = None
    if "context" in args.sim:
        weights, odds, n_starts = fit_hazard(inputs, params)
        print("hazard model on {} real gap starts, odds ratio per sd: {}".format(n_starts, odds))

    rows = []
    for sim in args.sim:
        for seed in range(args.seeds):
            rng = np.random.default_rng(seed)
            if sim == "transplant":
                atd, hidden = transplant(inputs, rng)
            else:
                atd, hidden = hide_ball(inputs.atd, rng, args.missing,
                                        lengths=None if sim == "uniform" else lengths,
                                        weights=weights if sim == "context" else None)
            res, _ = evaluate(masked_inputs(inputs, atd), params, gt_codes, gt, hidden, frozen_q)
            res.update({"sim": sim, "seed": seed, "frozen": bool(args.frozen),
                        "set": args.set or []})
            rows.append(res)
            missing = np.isnan(atd["ball"][PUBLIC[0]:PUBLIC[1], 0]).mean()
            res["missing_share"] = float(missing)
            print("{:10s} seed {}  spellF1 {:.4f}  macroF1 {:.4f}  poss {:>2}  | missing {:.0%}, "
                  "hidden {:>4} frames, gt D {:.0%}, team acc {:.3f}, D recall {}".format(
                      sim, seed, res["spellF1"], res["macroF1"], res["poss"], missing,
                      res["hidden_frames"], res["hidden_gt_dead_share"],
                      res["hidden_team_acc"] or float("nan"),
                      "{:.3f}".format(res["hidden_dead_recall"])
                      if res["hidden_dead_recall"] is not None else "-"))
            if args.json:
                with open(args.json, "a") as handle:
                    handle.write(json.dumps(res) + "\n")
    for sim in args.sim:
        s = np.array([r["spellF1"] for r in rows if r["sim"] == sim])
        m = np.array([r["macroF1"] for r in rows if r["sim"] == sim])
        print("{:10s} mean spellF1 {:.4f} (sd {:.4f})  macroF1 {:.4f}".format(
            sim, s.mean(), s.std(), m.mean()))


if __name__ == "__main__":
    main()
