"""Build a submission from the match data and score it on the public five minutes.

    python src/run.py                          # params from src/params.json
    python src/run.py --name try1 --set switch_cost=60 w_tag=0.5
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from possession.config import Params  # noqa: E402
from possession.paths import RESULTS_DIR, SRC  # noqa: E402
from possession import pipeline  # noqa: E402

DEFAULT_PARAMS = os.path.join(SRC, "params.json")


def parse_overrides(pairs):
    out = {}
    for pair in pairs or []:
        key, _, value = pair.partition("=")
        try:
            out[key] = int(value) if value.lstrip("-").isdigit() else float(value)
        except ValueError:
            out[key] = value
    return out


def report(result):
    print("spellF1 {:.4f}  macroF1 {:.4f}  possessions {} (gt {})  mean {:.1f}s (gt {:.1f}s)".format(
        result["spell_f1_mean"], result["macro_f1"], result["n_pred_spells"],
        result["n_gt_spells"], result["mean_spell_duration_s"],
        result["gt_mean_spell_duration_s"]))
    print("per class F1  " + "  ".join("{} {:.3f}".format(k, v["f1"])
                                       for k, v in result["per_class"].items()))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--params", default=DEFAULT_PARAMS)
    parser.add_argument("--set", nargs="+", metavar="KEY=VALUE", help="override params")
    parser.add_argument("--name", default="barca")
    parser.add_argument("--out", default=RESULTS_DIR)
    parser.add_argument("--refresh", action="store_true", help="re-read the raw ATD")
    args = parser.parse_args()

    params = Params.load(args.params if os.path.isfile(args.params) else None,
                         **parse_overrides(args.set))
    started = time.time()
    inputs = pipeline.Inputs(refresh=args.refresh)
    labels = pipeline.predict(inputs, params)
    path = pipeline.write(os.path.join(args.out, args.name + ".csv"), labels)
    print("wrote {} in {:.0f}s".format(os.path.relpath(path), time.time() - started))
    report(pipeline.score(labels))


if __name__ == "__main__":
    main()
