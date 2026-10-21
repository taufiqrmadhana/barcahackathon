"""Why each ground-truth possession of the public window was or was not recovered.

    python src/spells.py                          # the clean public window
    python src/spells.py --hide empirical --seeds 3   # leak-free masked runs (robustness.py)

Every GT possession spell gets one category, in this order of precedence:

  hit       a predicted spell of the same team has IoU >= 0.5 with it
  missed    less than 20% of it is predicted as its team
  split     two or more same-team predicted spells share it and none reaches IoU 0.5
  merged    the covering predicted spell also covers another GT spell of the same team, i.e. a
            possession in between was missed and the two were glued together
  boundary  found, alone, but it starts or ends too far away (offsets in frames, + = late)

Missed spells are then attributed by decoding the half again with the same emissions and
transition bonuses and a lower switch cost only:

  suppressed  found (covered >= 50%) at a lower switch cost: the evidence was there and the
              cost cancelled it
  dead        not found at any cost, and mostly predicted D
  no evidence not found at any cost: the tracking never supports that team there

The same table shows what the lower costs do to the whole window (possession count, spellF1), which
is the price of curing a miss by lowering the cost globally.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from possession.config import Params  # noqa: E402
from possession import hsmm, pipeline  # noqa: E402
from run import DEFAULT_PARAMS, parse_overrides  # noqa: E402

FPS = 25
COSTS = (16.0, 8.0, 4.0, 2.0, 0.5)
PUBLIC_HALF = 0


def segments(labels, lo, hi):
    x = labels[lo:hi]
    ch = np.flatnonzero(x[1:] != x[:-1]) + 1
    return [(lo + a, lo + b, x[a]) for a, b in zip(np.r_[0, ch], np.r_[ch, len(x)])]


def iou(a, b):
    inter = min(a[1], b[1]) - max(a[0], b[0])
    return max(0, inter) / float(max(a[1], b[1]) - min(a[0], b[0]))


def classify(gt_spells, pred_spells, labels):
    out = []
    for i, g in enumerate(gt_spells):
        a, b, team = g
        same = [p for p in pred_spells if p[2] == team and p[1] > a and p[0] < b]
        best = max([iou(g, p) for p in same], default=0.0)
        cover = float((labels[a:b] == team).mean())
        comp = {c: round(float((labels[a:b] == c).mean()), 2) for c in "HAD"
                if (labels[a:b] == c).any()}
        row = {"i": i, "start": a, "stop": b, "team": team, "dur": (b - a) / FPS,
               "iou": best, "cover": cover, "pred": comp, "offsets": None}
        if best >= 0.5:
            row["cat"] = "hit"
        elif cover < 0.2:
            row["cat"] = "missed"
        else:
            big = [p for p in same if min(p[1], b) - max(p[0], a) >= 0.2 * (b - a)]
            main = max(same, key=lambda p: min(p[1], b) - max(p[0], a))
            row["offsets"] = (int(main[0] - a), int(main[1] - b))
            others = [h for j, h in enumerate(gt_spells) if j != i and h[2] == team]
            glued = any(min(main[1], h[1]) - max(main[0], h[0]) >= 0.25 * (h[1] - h[0])
                        for h in others)
            row["cat"] = "split" if len(big) >= 2 else ("merged" if glued else "boundary")
        out.append(row)
    return out


def spurious(gt_spells, pred_spells, gt):
    out = {}
    for p in pred_spells:
        if max([iou(p, g) for g in gt_spells if g[2] == p[2]], default=0.0) >= 0.5:
            continue
        seg = gt[p[0]:p[1]]
        share = {c: (seg == c).mean() for c in "HAD"}
        if share["D"] >= 0.5:
            kind = "inside gt D"
        elif share[p[2]] < 0.5:
            kind = "wrong team"
        else:
            inside = [g for g in gt_spells if g[2] == p[2] and g[0] <= p[0] and p[1] <= g[1]]
            kind = "fragment of a gt spell" if inside else "spans several gt segments"
        out[kind] = out.get(kind, 0) + 1
    return out


def evidence_stats(dbg, a, b, team):
    side = 0 if team == "H" else 1
    e = dbg["emit"][a:b]
    own = e[:, :2].max(1) if team == "H" else e[:, 2:4].max(1)
    other = e[:, 2:4].max(1) if team == "H" else e[:, :2].max(1)
    delta = own - other
    best = cur = 0.0
    for d in delta:
        cur = max(0.0, cur + d)
        best = max(best, cur)
    ev = dbg["ev"]
    ctrl = ev["ctrl"][a:b, side]
    seen = ev["seen"][a:b]
    dist = ev["dist"][a:b]
    nearer = (dist[:, side] < dist[:, 1 - side])[seen]
    touch = dbg["events"]["touch"]
    inside = (touch["frame"] >= a - 12) & (touch["frame"] < b)
    count = ev["count"][a:b]
    return {"seen": float(seen.mean()),
            "ctrl>0.5": int((np.nan_to_num(ctrl) > 0.5).sum()),
            "nearer": float(nearer.mean()) if len(nearer) else float("nan"),
            "touches": "{}/{}".format(int((touch["team"][inside] == side).sum()), int(inside.sum())),
            "visible": "{}/{}".format(int(np.median(count[:, side])), int(np.median(count[:, 1 - side]))),
            "best_run": best}


def counterfactual(inputs, params, dbg, gt, gt_spells, gt_codes):
    blocks = [inputs.blocks[PUBLIC_HALF]]
    found = {}
    window = []
    for cost in COSTS:
        p = Params(**{**params.__dict__, "switch_cost": cost})
        labels, _ = hsmm.Model(p).decode(dbg["emit"], dbg["bonus"], blocks, inputs.n)
        r = pipeline.score(labels, gt_codes)
        window.append((cost, r["n_pred_spells"], r["spell_f1_mean"]))
        for i, (a, b, team) in enumerate(gt_spells):
            if i not in found and (labels[a:b] == team).mean() >= 0.5:
                found[i] = cost
    return found, window


def report(inputs, params, gt_codes, verbose=True, result=None):
    gt = np.array(["H", "A", "D", "X"])[gt_codes]
    labels, dbg = result or pipeline.predict(inputs, params, debug=True)
    idx = np.flatnonzero(gt != "X")
    lo, hi = idx[0], idx[-1] + 1
    gt_spells = [s for s in segments(gt, lo, hi) if s[2] in "HA"]
    pred_spells = [s for s in segments(labels, lo, hi) if s[2] in "HA"]
    rows = classify(gt_spells, pred_spells, labels)
    found, window = counterfactual(inputs, params, dbg, gt, gt_spells, gt_codes)
    for row in rows:
        row.update(evidence_stats(dbg, row["start"], row["stop"], row["team"]))
        if row["cat"] in ("missed", "boundary", "merged", "split"):
            if row["cat"] != "missed":
                row["why"] = "found at cost {:g}".format(found[row["i"]]) \
                    if row["i"] in found and found[row["i"]] < params.switch_cost else ""
            elif row["i"] in found:
                row["why"] = "suppressed (found at cost {:g})".format(found[row["i"]])
            elif row["pred"].get("D", 0) >= 0.5:
                row["why"] = "dead"
            else:
                row["why"] = "no evidence"
        else:
            row["why"] = ""
    if verbose:
        print("  #  team start  dur    cat      IoU  cover pred                      offsets      "
              "seen ctrl>.5 nearer touch vis  best_run  why")
        for r in rows:
            print("{:3d}  {}  {:6d} {:5.1f}s  {:8s} {:.2f} {:.2f}  {:25s} {:12s} {:.2f} {:6d}  "
                  "{:.2f}  {:5s} {:5s} {:7.1f}  {}".format(
                      r["i"], r["team"], r["start"], r["dur"], r["cat"], r["iou"], r["cover"],
                      str(r["pred"]), str(r["offsets"] or ""), r["seen"], r["ctrl>0.5"],
                      r["nearer"], r["touches"], r["visible"], r["best_run"], r["why"]))
        print("\nswitch cost, whole public half decoded with the same emissions:")
        for cost, n, f1 in window:
            print("  cost {:5.1f}  possessions {:3d} (gt {})  spellF1 {:.4f}".format(
                cost, n, len(gt_spells), f1))
        print("spurious predicted spells: {}".format(spurious(gt_spells, pred_spells, gt)))
    return rows, spurious(gt_spells, pred_spells, gt)


def summarise(all_rows):
    cats, whys = {}, {}
    for r in all_rows:
        cats[r["cat"]] = cats.get(r["cat"], 0) + 1
        if r["cat"] == "missed":
            key = r["why"].split(" (")[0]
            whys[key] = whys.get(key, 0) + 1
    fixable = [r for r in all_rows if r["cat"] != "hit" and "cost" in r["why"]]
    print("not hit but found at a lower switch cost: {} of {}".format(
        len(fixable), sum(r["cat"] != "hit" for r in all_rows)))
    short = [r for r in all_rows if r["dur"] < 4]
    print("categories {}  missed because {}".format(cats, whys))
    print("spells under 4 s: {}  of them hit {}".format(
        len(short), sum(r["cat"] == "hit" for r in short)))
    bounds = [r["offsets"] for r in all_rows if r["cat"] == "boundary"]
    if bounds:
        print("boundary offsets (frames, start/end): {}".format(bounds))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--params", default=DEFAULT_PARAMS)
    parser.add_argument("--set", nargs="+", metavar="KEY=VALUE")
    parser.add_argument("--hide", choices=["uniform", "empirical", "context"])
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--missing", type=float, default=0.41)
    args = parser.parse_args()

    params = Params.load(args.params if os.path.isfile(args.params) else None,
                         **parse_overrides(args.set))
    inputs = pipeline.Inputs()
    gt_codes = pipeline.public_gt()
    if not args.hide:
        rows, _ = report(inputs, params, gt_codes)
        print()
        summarise(rows)
        return

    import robustness
    lengths = None if args.hide == "uniform" else robustness.real_gaps(inputs)
    weights = robustness.fit_hazard(inputs, params)[0] if args.hide == "context" else None
    all_rows, extra = [], {}
    for seed in range(args.seeds):
        atd, _ = robustness.hide_ball(inputs.atd, np.random.default_rng(seed), args.missing,
                                      lengths, weights)
        print("\n=== {} seed {}".format(args.hide, seed))
        rows, sp = report(robustness.masked_inputs(inputs, atd), params, gt_codes)
        all_rows += rows
        for k, v in sp.items():
            extra[k] = extra.get(k, 0) + v
    print("\n=== all seeds")
    summarise(all_rows)
    print("spurious predicted spells: {}".format(extra))


if __name__ == "__main__":
    main()
