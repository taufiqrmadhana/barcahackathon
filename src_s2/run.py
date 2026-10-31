"""S2 pipeline: build a submission, score it on the public window, optionally sweep a parameter.

    python src_s2/run.py                                  # params from src_s2/params.json
    python src_s2/run.py --name s2_try --set min_run=12
    python src_s2/run.py --sweep min_run=1,4,8,12,20      # response curve, no CSV written
    python src_s2/run.py --spells --ref evaluation/results/taufiq_ref.csv   # per-GT-spell table
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import data  # noqa: E402
import s2  # noqa: E402
import possession_metrics as pm  # noqa: E402

GT = os.path.join(data.ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
OUT = os.path.join(data.ROOT, "evaluation", "results")


def parse(v):
    try:
        return int(v)
    except ValueError:
        return float(v)


def full_stats(lab):
    codes = pm.label_codes(lab)
    sp = pm.possession_spells(codes, data.HALVES)
    graded = np.concatenate([lab[a:b] for a, b in data.HALVES])
    return len(sp), float(np.mean(graded == "D")), np.mean([(b - a) / 25 for a, b, _ in sp])


def score(lab, gtc):
    r = pm.evaluate(gtc, pm.label_codes(lab), bootstrap_n=0)
    n, dead, mean = full_stats(lab)
    return r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"], n, dead, mean


def segs(l, lo, hi):
    ch = np.flatnonzero(l[lo + 1:hi] != l[lo:hi - 1]) + 1
    return [(lo + a, lo + b, l[lo + a]) for a, b in zip(np.r_[0, ch], np.r_[ch, hi - lo])]


def iou(a, b):
    return max(0, min(a[1], b[1]) - max(a[0], b[0])) / float(max(a[1], b[1]) - min(a[0], b[0]))


def spell_table(gt, preds):
    """Per GT possession: best same-team IoU for each prediction."""
    lo, hi = np.flatnonzero(gt != "X")[[0, -1]]
    hi += 1
    gts = [s for s in segs(gt, lo, hi) if s[2] in "HA"]
    print("  #  team  start   dur  " + "  ".join("{:>10s}".format(k[:10]) for k in preds))
    for i, g in enumerate(gts):
        cells = []
        for k, lab in preds.items():
            ps = [s for s in segs(lab, lo, hi) if s[2] == g[2]]
            best = max([iou(g, s) for s in ps], default=0)
            cells.append("{:10.2f}".format(best))
        print("{:3d}  {}  {:6d} {:5.1f}s ".format(i, g[2], g[0], (g[1] - g[0]) / 25) + "  ".join(cells))
    for k, lab in preds.items():
        print("  {}: {}".format(k, [(int(a), round((b - a) / 25, 1), str(c)) for a, b, c in segs(lab, lo, hi)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default=os.path.join(HERE, "params.json"))
    ap.add_argument("--set", nargs="+", default=[])
    ap.add_argument("--name", default="s2")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--sweep", nargs="+", default=[])
    ap.add_argument("--spells", action="store_true")
    ap.add_argument("--ref", default=os.path.join(OUT, "taufiq_ref.csv"))
    args = ap.parse_args()

    over = {k: parse(v) for k, v in (s.split("=") for s in args.set)}
    p = s2.Params.load(args.params, **over)
    inp = data.load()
    if p.get("shape_gap", 0) > 0 or p.get("shape_dead", 0) > 0:
        import shape
        inp["shape"] = shape.get_proba(inp, p)
    gt = pd.read_csv(GT)["label"].values.astype("<U1")
    gtc = pm.label_codes(gt)

    if args.sweep:
        for spec in args.sweep:
            key, vals = spec.split("=")
            for v in vals.split(","):
                q = s2.Params({**p, key: parse(v)})
                f, m, n, nf, dead, mean = score(s2.LABELS[s2.predict(inp, q)], gtc)
                print("{}={:<6} spellF1 {:.4f} macroF1 {:.4f} poss {:2d} | full poss {} dead {:.3f} "
                      "mean {:.1f}s".format(key, v, f, m, n, nf, dead, mean))
        return

    lab = LABELS_FROM = s2.LABELS[np.asarray(s2.predict(inp, p))]
    path = os.path.join(args.out, args.name + ".csv")
    os.makedirs(args.out, exist_ok=True)
    pd.DataFrame({"frame": np.arange(len(lab)), "label": lab}).to_csv(path, index=False)
    f, m, n, nf, dead, mean = score(lab, gtc)
    print("wrote {}\npublic spellF1 {:.4f} macroF1 {:.4f} poss {} (gt 17) | full poss {} dead {:.3f} "
          "mean {:.1f}s".format(path, f, m, n, nf, dead, mean))
    if args.spells:
        preds = {"s2": LABELS_FROM}
        if os.path.isfile(args.ref):
            preds["ref"] = pd.read_csv(args.ref)["label"].values.astype("<U1")
        spell_table(gt, preds)


if __name__ == "__main__":
    main()
