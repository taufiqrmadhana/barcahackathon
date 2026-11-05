"""Leak-free hidden-ball test: transplant real ball-missing masks from other 5-minute windows of
the match onto the public window, hide the ball there BEFORE any processing, rerun, score.

    python src_s2/hidden.py [--set k=v ...] [--n 19]

The mask of a source window is "ball NaN in the raw ATD" for each frame; it is copied frame by
frame onto the public window (same offset). Only the ATD ball of the public window is changed;
labels never enter the pipeline. Caveat: transplanted gaps are not aligned with the public
window's real stoppages, so this is pessimistic for the "missing ball = stoppage" cue and
neutral for the owner logic.
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
import importlib.util as _u  # noqa: E402
_spec = _u.spec_from_file_location("s2run", os.path.join(HERE, "run.py"))
s2run = _u.module_from_spec(_spec)
_spec.loader.exec_module(s2run)
GT, parse = s2run.GT, s2run.parse

PUB = (11185, 18685)
CROP = (0, PUB[1] + 3000)  # the pipeline is local; crop for speed (checked: identical clean score)


def source_windows():
    w = PUB[1] - PUB[0]
    out = []
    for a, b in data.HALVES:
        s = PUB[1] if a == PUB[0] else a
        while s + w <= b:
            out.append(s)
            s += w
    return out


def crop(inp, lo, hi):
    o = dict(inp)
    for k in ("px", "py", "pv", "bx", "by", "shape"):
        if inp.get(k) is not None:
            o[k] = inp[k][lo:hi].copy()
    o["n"] = hi - lo
    return o


def score_window(lab, gtc):
    r = pm.evaluate(gtc, pm.label_codes(lab), bootstrap_n=0)
    return r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"]


def run(p, inp_full, gtc_full, starts, predict=None, verbose=True):
    predict = predict or s2.predict
    lo, hi = CROP
    inp = crop(inp_full, lo, hi)
    gtc = gtc_full[lo:hi]
    clean = score_window(s2.LABELS[predict(inp, p)], gtc)
    rows = []
    w = PUB[1] - PUB[0]
    raw_missing = np.isnan(inp_full["bx"])
    for s in starts:
        m = raw_missing[s:s + w]
        q = crop(inp_full, lo, hi)
        a, b = PUB[0] - lo, PUB[1] - lo
        q["bx"][a:b][m] = np.nan
        q["by"][a:b][m] = np.nan
        f, mf, n = score_window(s2.LABELS[predict(q, p)], gtc)
        miss = float(np.mean(np.isnan(q["bx"][a:b])))
        rows.append((s, miss, f, mf, n))
        if verbose:
            print("  src {:6d} ball-missing {:.2f}  spellF1 {:.4f} macroF1 {:.4f} poss {}".format(s, miss, f, mf, n))
    arr = np.array([r[2] for r in rows])
    mac = np.array([r[3] for r in rows])
    print("clean spellF1 {:.4f} macroF1 {:.4f} poss {} | hidden ({} masks) spellF1 mean {:.4f} sd {:.4f} "
          "min {:.4f} macroF1 mean {:.4f} poss mean {:.1f}".format(
              clean[0], clean[1], clean[2], len(rows), arr.mean(), arr.std(), arr.min(), mac.mean(),
              np.mean([r[4] for r in rows])))
    return clean, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default=os.path.join(HERE, "params.json"))
    ap.add_argument("--set", nargs="+", default=[])
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--sweep", default=None, help="key=v1,v2,... (clean + hidden per value)")
    args = ap.parse_args()
    over = {k: parse(v) for k, v in (s.split("=") for s in args.set)}
    p = s2.Params.load(args.params, **over)
    inp = data.load()
    if p.get("shape_gap", 0) > 0 or p.get("shape_dead", 0) > 0 or (args.sweep or "").startswith("shape"):
        import shape
        inp["shape"] = shape.get_proba(inp, p)  # players only: identical under any ball mask
    gtc = pm.label_codes(pd.read_csv(GT)["label"].values.astype("<U1"))
    if args.sweep:
        key, vals = args.sweep.split("=")
        for v in vals.split(","):
            print("{}={}: ".format(key, v), end="")
            run(s2.Params({**p, key: parse(v)}), inp, gtc, source_windows(), verbose=False)
        return
    run(p, inp, gtc, source_windows(), verbose=not args.quiet)


if __name__ == "__main__":
    main()
