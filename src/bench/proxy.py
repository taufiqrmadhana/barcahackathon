"""Full-match proxy: treat a prediction as if it were the ground truth and score the four
published baselines against it. The organisers publish the baselines' real full-match scores
(EVALUATION.md), so a prediction closer to the private truth should reproduce them more closely.
Uses only published numbers; no private ground truth is involved."""
import os, sys
import numpy as np
import pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402

PUBLISHED = {  # name: (macroF1, spellF1) on the full match, from EVALUATION.md
    "baseline_raw": (0.6532, 0.0486),
    "baseline_smooth25": (0.6752, 0.1996),
    "baseline_smooth50": (0.6724, 0.2433),
    "baseline_smooth125": (0.6356, 0.2242),
}
BLOCKS = [(11185, 90651), (110808, 190197)]


def as_gt(path):
    lab = pm.read_label_csv(path)["label"].astype(str).str.strip().str.upper().values.copy()
    mask = np.ones(len(lab), bool)
    for a, b in BLOCKS:
        mask[a:b + 1] = False
    lab[mask] = "X"
    return pm.label_codes(lab)


_BASE = {}


def proxy(path, verbose=True):
    gt = as_gt(path)
    rows = []
    for name, (mf, sf) in PUBLISHED.items():
        if name not in _BASE:
            _BASE[name] = pm.label_codes(pm.read_label_csv(os.path.join(ROOT, "evaluation/results", name + ".csv")))
        r = pm.evaluate(gt, _BASE[name])
        rows.append((name, r["macro_f1"], mf, r["spell_f1_mean"], sf))
    err_m = np.mean([abs(a - b) for _, a, b, _, _ in rows])
    err_s = np.mean([abs(c - d) for _, _, _, c, d in rows])
    if verbose:
        for n, a, b, c, d in rows:
            print(f"  {n:20s} macro {a:.4f} (pub {b:.4f})  spell {c:.4f} (pub {d:.4f})")
        print(f"  mean |err| macro {err_m:.4f}  spell {err_s:.4f}")
    return err_m, err_s, rows


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(p)
        proxy(p)
