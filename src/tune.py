"""Grid search over decoder params, scored on the public five minutes.

Seventeen possessions cannot rank anything finely (EVALUATION.md): treat the top of this table as
a neighbourhood worth checking, not as the answer. Every row also shows full-match statistics
(possession count, dead share, dead stretches), which a good config should keep plausible.

The shape model is fitted once with the base params and frozen, so only the decode is repeated.

    python src/tune.py --grid switch_cost=10,20,40 dead_entry_cost=10,30
"""

import argparse
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from possession.config import Params  # noqa: E402
from possession import events, fuse, hsmm, pipeline  # noqa: E402
from run import DEFAULT_PARAMS  # noqa: E402

DEFAULT_GRID = ["switch_cost=10,20,30", "dead_entry_cost=10,20,30", "w_shape_known=0,0.2"]


def parse_grid(items):
    grid = {}
    for item in items:
        key, _, values = item.partition("=")
        grid[key] = [int(v) if v.lstrip("-").isdigit() else
                     (float(v) if v.replace(".", "", 1).lstrip("-").isdigit() else v)
                     for v in values.split(",")]
    return grid


def decode_with(inputs, params, q):
    trk, ev = inputs.tracking(params), inputs.evidence(params)
    evs = events.build(trk, ev, inputs.tags, inputs.blocks, params)
    evs["tag_starts"] = inputs.tags["sp_start"]
    emit, _ = fuse.emissions(ev, q, inputs.tags, evs, params)
    labels, _ = hsmm.Model(params).decode(emit, fuse.bonuses(evs, inputs.n, params),
                                          inputs.blocks, inputs.n)
    return labels


def full_stats(labels, inputs):
    graded = labels[inputs.graded]
    change = np.flatnonzero(graded[1:] != graded[:-1]) + 1
    starts = np.r_[0, change]
    kinds = graded[starts]
    return int((kinds != "D").sum()), float((graded == "D").mean()), int((kinds == "D").sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--grid", nargs="+", default=DEFAULT_GRID, metavar="KEY=V1,V2")
    parser.add_argument("--params", default=DEFAULT_PARAMS)
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    grid = parse_grid(args.grid)
    base_path = args.params if os.path.isfile(args.params) else None
    inputs = pipeline.Inputs()
    gt = pipeline.public_gt()
    _, dbg = pipeline.predict(inputs, Params.load(base_path), debug=True)
    q = dbg["shape"]

    rows = []
    for combo in itertools.product(*grid.values()):
        overrides = dict(zip(grid, combo))
        params = Params.load(base_path, **overrides)
        labels = decode_with(inputs, params, q)
        result = pipeline.score(labels, gt)
        rows.append((result["spell_f1_mean"], result["macro_f1"], result["n_pred_spells"],
                     full_stats(labels, inputs), overrides))

    rows.sort(key=lambda row: (-row[0], -row[1]))
    print("spellF1  macroF1  poss  | full: poss  dead  stretches | params")
    for spell, macro, n, (fp, fd, fs), overrides in rows[:args.top]:
        print("{:.4f}   {:.4f}   {:>4}  |      {:>4}  {:.3f}  {:>4}      | {}".format(
            spell, macro, n, fp, fd, fs,
            " ".join("{}={}".format(k, v) for k, v in overrides.items())))


if __name__ == "__main__":
    main()
