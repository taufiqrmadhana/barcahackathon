"""Fixed-config public sanity check, including leak-free masked runs. Aggregates only.

Variant names follow shape_bench.py: B_ball adds supervised ownership/transition evidence,
S_shape replaces the final H/A shape ratio, BS_both does both. S_* rows need --shape-model.
"""

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from possession import pipeline, ownership, supervised_shape, fuse, hsmm
from possession.config import Params
from possession.paths import ROOT
from bench.ownership_bench import metrics, sha
from robustness import transplant, masked_inputs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=str(Path(ROOT, "pretrain_data/ownership_v1/ownership_all.joblib")))
    ap.add_argument("--shape-model", default="")
    ap.add_argument("--name", default="ownership_v1", help="submission CSV name for the clean run")
    ap.add_argument("--out", default=str(Path(ROOT, "cache/ownership_v1/public_sanity.json")))
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1])
    args = ap.parse_args()
    p = Params.load(str(Path(ROOT, "src/params.json")))
    bp = replace(p, ownership_model=args.model, supervised_shape_model=args.shape_model)
    base = pipeline.Inputs()
    gt = np.array(["H", "A", "D", "X"])[pipeline.public_gt()]
    rows = []
    for seed in [-1] + args.seeds:
        if seed == -1:
            inputs = base
        else:
            atd, _ = transplant(base, np.random.default_rng(seed))
            inputs = masked_inputs(base, atd)
        print("decode", "clean" if seed == -1 else "transplant", seed, flush=True)
        baseline, dbg = pipeline.predict(inputs, p, debug=True)
        op = ownership.infer(inputs, bp)
        decode = lambda emit, bonus: hsmm.Model(bp).decode(emit, bonus, inputs.blocks, inputs.n)[0]
        variants = {"A": baseline,
                    "B_ball": decode(*ownership.apply(dbg["emit"], dbg["bonus"], op, bp))}
        if args.shape_model:
            q = supervised_shape.replace_team(dbg["shape"], supervised_shape.infer(inputs, bp))
            se, pd = fuse.emissions(dbg["ev"], q, inputs.tags, dbg["events"], p)
            np.testing.assert_array_equal(pd, dbg["p_dead"])
            variants["S_shape"] = decode(se, dbg["bonus"])
            variants["BS_both"] = decode(*ownership.apply(se, dbg["bonus"], op, bp))
        if seed == -1:
            full = variants["BS_both" if args.shape_model else "B_ball"]
            production = pipeline.predict(inputs, bp)
            if not np.array_equal(full, production):
                raise AssertionError("production differs from frozen-shape experiment")
            pipeline.write(str(Path(ROOT, f"evaluation/results/{args.name}.csv")), full)
            print("production/manual parity: exact; submission written", flush=True)
        for name, lab in variants.items():
            row = {"variant": name, "seed": seed,
                   "condition": "clean" if seed == -1 else "transplant",
                   **metrics(lab, gt, dbg["ev"]["seen"], baseline)}
            rows.append(row)
            print(json.dumps(row), flush=True)
        # Save progress without any frame-level data.
        output = {"sanity_only": True, "selected_on_public": False,
                  "checkpoint_sha256": sha(args.model),
                  "shape_checkpoint_sha256": sha(args.shape_model) if args.shape_model else None,
                  "production_parity": True, "results": rows}
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
