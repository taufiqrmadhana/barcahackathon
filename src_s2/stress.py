"""Noise response curves: score param variants under iid Gaussian ball noise (in-process, shape
probabilities computed once; they use players only, so they do not depend on the ball).

    python src_s2/stress.py --params src_s2/params_s7b.json --variants v.json --sigmas 0 0.1 0.2 0.3 --seeds 0 1 2
v.json = {"name": {"key": value, ...}, ...}; "base" (no overrides) is always first.
Prints per variant and sigma: public spellF1 mean/min over seeds, pooled spellF1 mean, full poss.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE)))), "personal", "loop"))
import data  # noqa: E402
import s2  # noqa: E402
import shape  # noqa: E402
import loopmetrics as lm  # noqa: E402
import possession_metrics as pm  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True)
    ap.add_argument("--variants", default=None)
    ap.add_argument("--sigmas", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.3])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args()
    base = s2.Params.load(a.params)
    variants = {"base": {}}
    if a.variants:
        variants.update(json.load(open(a.variants)))
    inp = data.load()
    inp["shape"] = shape.get_proba(inp, base)
    tsets = lm.truth_sets()
    for name, over in variants.items():
        for sg in a.sigmas:
            pubs, pools, ns = [], [], []
            for sd in (a.seeds if sg > 0 else [0]):
                p = s2.Params({**base, **over, "noise_sigma": sg, "noise_seed": sd})
                lab = s2.LABELS[s2.predict(inp, p)]
                pred = pm.label_codes(lab)
                tabs = [lm.match_table(t, pred) for _, t in tsets]
                pubs.append(pm.evaluate(tsets[0][1], pred, bootstrap_n=0)["spell_f1_mean"])
                pools.append(lm.pooled_f1(tabs))
                graded = np.concatenate([lab[x:y] for x, y in data.HALVES])
                ns.append((len(pm.possession_spells(pred, data.HALVES)), float(np.mean(graded == "D"))))
            print("{:<22s} sigma {:.2f} | public {:.4f} (min {:.4f}) | pooled {:.4f} | full poss {:.0f} dead {:.3f}".format(
                name, sg, np.mean(pubs), np.min(pubs), np.mean(pools), np.mean([n[0] for n in ns]), np.mean([n[1] for n in ns])), flush=True)


if __name__ == "__main__":
    main()
