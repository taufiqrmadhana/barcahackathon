"""Grid search over a few params, scored on the public five minutes.

Seventeen possessions cannot rank anything finely (EVALUATION.md): treat the top of this table as
a neighbourhood worth checking by eye, not as the answer.

    python src/tune.py --grid switch_cost=10,20,40 dead_cost=10,30 w_tag=0,0.3
"""

import argparse
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from possession.config import Params  # noqa: E402
from possession import pipeline  # noqa: E402
from run import DEFAULT_PARAMS  # noqa: E402

DEFAULT_GRID = ["switch_cost=10,20,30,40", "dead_cost=10,20,30", "w_tag=0,0.3,0.6"]


def parse_grid(items):
    grid = {}
    for item in items:
        key, _, values = item.partition("=")
        grid[key] = [int(v) if v.lstrip("-").isdigit() else float(v) for v in values.split(",")]
    return grid


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--grid", nargs="+", default=DEFAULT_GRID, metavar="KEY=V1,V2")
    parser.add_argument("--params", default=DEFAULT_PARAMS)
    parser.add_argument("--top", type=int, default=15)
    args = parser.parse_args()

    grid = parse_grid(args.grid)
    base = args.params if os.path.isfile(args.params) else None
    inputs = pipeline.Inputs()
    gt = pipeline.public_gt()

    rows = []
    for combo in itertools.product(*grid.values()):
        overrides = dict(zip(grid, combo))
        result = pipeline.score(pipeline.predict(inputs, Params.load(base, **overrides)), gt)
        rows.append((result["spell_f1_mean"], result["macro_f1"], result["n_pred_spells"],
                     overrides))

    rows.sort(key=lambda row: (-row[0], -row[1]))
    print("spellF1  macroF1  poss  params")
    for spell, macro, n, overrides in rows[:args.top]:
        print("{:.4f}   {:.4f}   {:>4}  {}".format(
            spell, macro, n, " ".join("{}={}".format(k, v) for k, v in overrides.items())))


if __name__ == "__main__":
    main()
