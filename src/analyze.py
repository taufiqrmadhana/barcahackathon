"""Error analysis: where the pipeline goes wrong, in aggregate. Prints numbers, never tracking.

    python src/analyze.py                   # params from src/params.json
    python src/analyze.py --set switch_cost=30 --segments

Public window: score of both decodes, per-segment composition, boundary offsets.
Full match (no ground truth): possession count and length, dead-ball share, agreement with Smart
Tagging set pieces and phases, and how the shape model does on frames where the ball is hidden
on purpose.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from possession.config import Params  # noqa: E402
from possession import events as events_mod, pipeline  # noqa: E402
from run import DEFAULT_PARAMS, parse_overrides  # noqa: E402

FPS = 25


def segments(labels, lo, hi):
    x = labels[lo:hi]
    if not len(x):
        return []
    ch = np.flatnonzero(x[1:] != x[:-1]) + 1
    return [(lo + a, lo + b, x[a]) for a, b in zip(np.r_[0, ch], np.r_[ch, len(x)])]


def spells(labels, blocks):
    out = []
    for _, first, last in blocks:
        out += [s for s in segments(labels, first, last + 1) if s[2] in "HA"]
    return out


def public_report(labels, gt, show_segments):
    idx = np.flatnonzero(gt != "X")
    lo, hi = idx[0], idx[-1] + 1
    gt_segs = segments(gt, lo, hi)
    if show_segments:
        print("\nground-truth segments vs prediction")
        for a, b, k in gt_segs:
            comp = {c: int((labels[a:b] == c).sum()) for c in "HAD" if (labels[a:b] == c).any()}
            print("  {:6d}-{:6d} {} {:5.1f}s  pred {}".format(a, b, k, (b - a) / FPS, comp))
    gt_bounds = [s[0] for s in gt_segs[1:]]
    pred_bounds = np.array([s[0] for s in segments(labels, lo, hi)[1:]])
    offsets = np.array([pred_bounds[np.argmin(np.abs(pred_bounds - f))] - f
                        for f in gt_bounds]) if len(pred_bounds) else np.array([10 ** 6])
    within = lambda t: int((np.abs(offsets) <= t).sum())  # noqa: E731
    print("boundaries  gt {}  pred {}  within 0.5s {}  within 1s {}  within 2s {}".format(
        len(gt_bounds), len(pred_bounds), within(12), within(25), within(50)))


def full_report(labels, dbg, inputs):
    blocks = inputs.blocks
    graded = inputs.graded
    sp = spells(labels, blocks)
    lengths = np.array([b - a for a, b, _ in sp]) / FPS
    dead = segments(labels, 0, len(labels))
    dead = [s for s in dead if s[2] == "D"]
    print("\nfull match (no ground truth)")
    print("  possessions {}  mean {:.1f}s  median {:.1f}s  <2s {}  dead share {:.3f}  "
          "dead stretches {}".format(len(sp), lengths.mean(), np.median(lengths),
                                     int((lengths < 2).sum()), (labels[graded] == "D").mean(),
                                     len(dead)))
    seen = dbg["ev"]["seen"]
    print("  ball seen {:.3f}  possessions decided with no ball at all {}".format(
        seen[graded].mean(), sum(1 for a, b, _ in sp if not seen[a:b].any())))

    tags = inputs.tags
    starts = [f for f in tags["sp_start"] if graded[f]]
    hit = 0
    for f in starts:
        window = labels[f:min(len(labels), f + 250)]
        hit += ("D" in window[:150])
    print("  tagged set pieces with a D in the 6 s after the tag starts: {}/{}".format(
        hit, len(starts)))
    team = tags["team"]
    alive = graded & np.isin(labels, ["H", "A"]) & (team != 0)
    agree = np.where(team[alive] > 0, "H", "A") == labels[alive]
    print("  Smart Tagging phase team agrees on {:.3f} of tagged alive frames".format(
        agree.mean()))
    print("  events {}".format(events_mod.summary(dbg["events"])))


def shape_report(dbg, gt, labels1):
    q = dbg["shape"]
    graded = gt != "X"
    alive = graded & (gt != "D")
    team = np.where(q[:, 0] >= q[:, 1], "H", "A")
    print("\nshape model (players only) on the public window")
    print("  team accuracy on alive frames {:.3f}".format((team[alive] == gt[alive]).mean()))
    d = q[:, 2]
    if graded.any():
        print("  mean P(D)  gt D {:.3f}  gt alive {:.3f}".format(d[graded & (gt == "D")].mean(),
                                                               d[alive].mean()))
    print("  agreement with decode #1 where the ball is tracked: {:.3f}".format(
        (team[dbg["train"]] == labels1[dbg["train"]]).mean()))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--params", default=DEFAULT_PARAMS)
    parser.add_argument("--set", nargs="+", metavar="KEY=VALUE")
    parser.add_argument("--segments", action="store_true")
    args = parser.parse_args()

    params = Params.load(args.params if os.path.isfile(args.params) else None,
                         **parse_overrides(args.set))
    inputs = pipeline.Inputs()
    labels, dbg = pipeline.predict(inputs, params, debug=True)
    gt_codes = pipeline.public_gt()
    gt = np.array(["H", "A", "D", "X"])[gt_codes]

    for name, lab in (("decode #1 (ball only)", dbg["labels1"]), ("final", labels)):
        r = pipeline.score(lab, gt_codes)
        print("{:22s} spellF1 {:.4f}  macroF1 {:.4f}  poss {} (gt {})  F1 H {:.3f} A {:.3f} "
              "D {:.3f}".format(name, r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"],
                                r["n_gt_spells"], r["per_class"]["H"]["f1"],
                                r["per_class"]["A"]["f1"], r["per_class"]["D"]["f1"]))
    public_report(labels, gt, args.segments)
    shape_report(dbg, gt, dbg["labels1"])
    full_report(labels, dbg, inputs)


if __name__ == "__main__":
    main()
