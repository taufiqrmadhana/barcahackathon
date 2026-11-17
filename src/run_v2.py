"""v2: build a submission from TXT + XML + homography + Smart Tagging, then evaluate it.

    python src/run_v2.py                                   # params_v2.json -> results/barca_v2.csv
    python src/run_v2.py --set w_memory=0.8 --name try     # one-off override
    python src/run_v2.py --bootstrap 1000                  # with the segment-bootstrap interval
    python src/run_v2.py --grid switch_cost=20,30 w_memory=0.3,0.6   # small sweep, no file written

Scores against the public five minutes with the official metric code (possession_metrics), and
prints whole-match possession counts, which need no ground truth and catch over-fragmenting.
Run evaluation/run_evaluation.py afterwards for the full official board.
"""

import argparse
import itertools
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from possession.config_valdi import ParamsV2  # noqa: E402
from possession.paths import RESULTS_DIR, SRC  # noqa: E402
from possession import pipeline_v2 as pipeline  # noqa: E402
from run import parse_overrides, report  # noqa: E402
# pyrefly: ignore[missing-import]
import possession_metrics as pm  # noqa: E402

V1_PARAMS = os.path.join(SRC, "params.json")
V2_PARAMS = os.path.join(SRC, "params_v2.json")


def load_params(path, overrides):
    if path and os.path.isfile(path):
        return ParamsV2.load(path, **overrides)
    # first run: start from the tuned v1 values, v2 defaults for the rest
    return ParamsV2.load(V1_PARAMS if os.path.isfile(V1_PARAMS) else None, **overrides)


def sweep(inputs, args, gt):
    grid = {}
    for item in args.grid:
        key, _, values = item.partition("=")
        grid[key] = [parse_overrides(["x=" + v])["x"] for v in values.split(",")]
    base = parse_overrides(args.set)
    rows = []
    for combo in itertools.product(*grid.values()):
        overrides = dict(base, **dict(zip(grid, combo)))
        labels = pipeline.predict(inputs, load_params(args.params, overrides))
        r = pipeline.score(labels, gt)
        summary = pipeline.match_summary(labels, inputs.blocks)
        full = sum(s["possessions"] for s in summary)
        dead = sum(s["share_D"] for s in summary) / len(summary)
        rows.append((r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"], full, dead, overrides))
    rows.sort(key=lambda row: (-row[0], -row[1]))
    print("spellF1  macroF1  poss(pub)  poss(match)  D(match)  params")
    for spell, macro, n, full, dead, ov in rows[:args.top]:
        print("{:.4f}   {:.4f}   {:>9}  {:>11}  {:>7.1%}  {}".format(
            spell, macro, n, full, dead, " ".join("{}={}".format(k, ov[k]) for k in grid)))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--params", default=V2_PARAMS)
    parser.add_argument("--set", nargs="+", metavar="KEY=VALUE", help="override params")
    parser.add_argument("--grid", nargs="+", metavar="KEY=V1,V2", help="sweep instead of write")
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument("--name", default="barca_v2")
    parser.add_argument("--out", default=RESULTS_DIR)
    parser.add_argument("--bootstrap", type=int, default=0)
    parser.add_argument("--save-params", action="store_true",
                        help="write the effective params to src/params_v2.json")
    parser.add_argument("--refresh", action="store_true", help="re-read the raw ATD")
    args = parser.parse_args()

    started = time.time()
    inputs = pipeline.Inputs(refresh=args.refresh)
    gt = pipeline.public_gt()
    if args.grid:
        sweep(inputs, args, gt)
        return

    params = load_params(args.params, parse_overrides(args.set))
    labels = pipeline.predict(inputs, params)
    path = pipeline.write(os.path.join(args.out, args.name + ".csv"), labels)
    print("wrote {} in {:.0f}s".format(os.path.relpath(path), time.time() - started))
    if args.save_params:
        params.save(V2_PARAMS)
        print("saved {}".format(os.path.relpath(V2_PARAMS)))

    cam = inputs.camera(params.err_scale, params.pan_window)
    print("metadata  {} (home) vs {}  fps {}  pitch {}  home attacks {} in 1st half".format(
        inputs.meta.team_names[0], inputs.meta.team_names[1], inputs.meta.fps,
        inputs.meta.pitch_m, "left->right" if inputs.meta.home_attack_first > 0 else "right->left"))
    print("camera    image {}x{}  usable matrices {:.1%}".format(
        cam.image_size[0], cam.image_size[1], cam.ok.mean()))

    print("\npublic window (5 min):")
    result = pm.evaluate(gt, pm.label_codes(labels), bootstrap_n=args.bootstrap)
    report(result)
    if args.bootstrap:
        for key in ("macro_f1_ci", "spell_f1_ci"):
            if key in result:
                print("{} {}".format(key, result[key]))

    print("\nwhole match (no ground truth needed):")
    for row in pipeline.match_summary(labels, inputs.blocks):
        print("  half {half}: {possessions} possessions, mean {mean_s:.1f}s, "
              "H {share_H:.1%}  A {share_A:.1%}  D {share_D:.1%}".format(**row))


if __name__ == "__main__":
    main()
