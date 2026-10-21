"""How the pipeline holds up when the ball disappears the way it does in the rest of the match.

The public window has the ball 88% of the time, the full match 59%, almost always in gaps of
3 s or more. This hides the ball on random gaps inside the public window until a target share is
missing, reruns the decode, and scores it. The shape model is fitted once on the untouched data
and frozen (its own features never use the ball).

    python src/robustness.py --missing 0.41 --seeds 5
"""

import argparse
import copy
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from possession.config import Params  # noqa: E402
from possession import pipeline  # noqa: E402
from run import DEFAULT_PARAMS, parse_overrides  # noqa: E402
from tune import decode_with  # noqa: E402

PUBLIC = (11185, 18685)


def hide_ball(atd, rng, share, lo, hi, min_gap=75, max_gap=1000):
    ball = atd["ball"].copy()
    seen = ~np.isnan(ball[lo:hi, 0])
    target = share * (hi - lo) - (~seen).sum()
    while target > 0:
        length = int(min(max_gap, np.exp(rng.uniform(np.log(min_gap), np.log(max_gap)))))
        start = int(rng.integers(lo, hi - length))
        newly = (~np.isnan(ball[start:start + length, 0])).sum()
        ball[start:start + length] = np.nan
        target -= newly
    out = dict(atd)
    out["ball"] = ball
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--params", default=DEFAULT_PARAMS)
    parser.add_argument("--set", nargs="+", metavar="KEY=VALUE")
    parser.add_argument("--missing", type=float, default=0.41)
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()

    params = Params.load(args.params if os.path.isfile(args.params) else None,
                         **parse_overrides(args.set))
    inputs = pipeline.Inputs()
    gt = pipeline.public_gt()
    labels, dbg = pipeline.predict(inputs, params, debug=True)
    q = dbg["shape"]
    clean = pipeline.score(labels, gt)
    print("no ball hidden      spellF1 {:.4f}  macroF1 {:.4f}  poss {}".format(
        clean["spell_f1_mean"], clean["macro_f1"], clean["n_pred_spells"]))

    spell, macro = [], []
    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)
        masked = copy.copy(inputs)
        masked.atd = hide_ball(inputs.atd, rng, args.missing, *PUBLIC)
        masked._cache = {}
        result = pipeline.score(decode_with(masked, params, q), gt)
        spell.append(result["spell_f1_mean"])
        macro.append(result["macro_f1"])
        print("seed {}  hidden {:.0%}  spellF1 {:.4f}  macroF1 {:.4f}  poss {}".format(
            seed, args.missing, spell[-1], macro[-1], result["n_pred_spells"]))
    print("mean                spellF1 {:.4f}  macroF1 {:.4f}".format(np.mean(spell),
                                                                       np.mean(macro)))


if __name__ == "__main__":
    main()
