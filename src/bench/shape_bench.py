"""Three-game LOGO: isolate supervised shape from learned ball ownership.

All hyperparameters and decoder weights remain fixed. Public competition labels are
never loaded. One ambiguous event timestamp in game 3 is masked in training only.
"""
import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path
import sys

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench import metrica_open as metrica
from bench.ownership_bench import audit, inputs_for, metrics, sha, SEED
from possession import atd as atd_io, pipeline, ownership, supervised_shape, fuse, hsmm
from possession.config import Params
from possession.paths import ROOT


def training_labels(game, n):
    labels = metrica.labels(game, n)
    events = metrica.events_table(game)
    ev = events[events.Type.isin(metrica.POSSESSION)]
    count = ev.groupby(["Period", "Start Frame"]).Team.nunique()
    for _, f in count[count > 1].index:
        frame = int(f) - 1
        labels[max(0, frame - 25):min(n, frame + 26)] = "X"
    return labels


def train(games, p, ball):
    owner_batches, shape_batches = [], []
    for game in games:
        for condition in ("clean", "corrupt"):
            inputs = inputs_for(game, condition, SEED + game, ball)
            labels = training_labels(game, inputs.n)
            views, support, names = ownership.features(inputs.tracking(p), inputs.blocks)
            owner_batches.append((views, support, names, labels, inputs.blocks))
            x, names = supervised_shape.features(inputs, p)
            shape_batches.append((x, names, labels))
    owner = ownership.fit(owner_batches, p, games)
    del owner_batches
    gc.collect()
    shape = supervised_shape.fit(shape_batches, p, games)
    return owner, shape


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(Path(ROOT, "cache/ownership_v2/logo")))
    ap.add_argument("--games", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--conditions", nargs="+", choices=["clean", "atd", "corrupt"],
                    default=["clean", "atd", "corrupt"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--fit-all", action="store_true")
    ap.add_argument("--params", default=str(Path(ROOT, "src/configs/external_training.json")))
    args = ap.parse_args()
    if len(set(args.games)) != len(args.games) or len(args.games) < 2:
        raise ValueError("supply at least two distinct games")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    p = Params.load(args.params, pre_kickoff_dead=0, goal_dead=0)
    if p.ownership_model or p.supervised_shape_model or p.w_near or p.w_near_switch:
        raise ValueError("baseline configuration must not include experimental models")
    audits = [audit(game) for game in args.games]
    manifest = {"params": asdict(p), "seed": SEED, "games": args.games,
                "eval_seeds": args.seeds, "conditions": args.conditions, "fit_all": args.fit_all,
                "label_convention": "metrica-events-v1; training excludes +/-1 s around contradictory owner events",
                "variants": ["A", "B_ball", "S_shape", "BS_both"],
                "source_hashes": {str(f.relative_to(ROOT)): sha(f)
                    for folder in ("src/possession", "src/bench")
                    for f in sorted(Path(ROOT, folder).glob("*.py"))}, "data": audits}
    target = out / "manifest.json"
    if target.exists() and json.loads(target.read_text()) != manifest:
        raise ValueError("use a fresh output directory for a changed experiment")
    target.write_text(json.dumps(manifest, indent=2) + "\n")
    ball = atd_io.load()["ball"]
    with (out / "results.jsonl").open("w") as log:
        for held in ([None] if args.fit_all else args.games):
            games = [g for g in args.games if g != held]
            print("train", games, "held out", held, flush=True)
            owner, shape = train(games, p, ball)
            owner["experiment"] = shape["experiment"] = manifest
            suffix = "all" if held is None else f"holdout_{held}"
            joblib.dump(owner, out / f"ownership_{suffix}.joblib")
            joblib.dump(shape, out / f"shape_{suffix}.joblib")
            if held is None:
                print("saved all-game checkpoints", flush=True)
                continue
            assert held not in owner["train_games"] and held not in shape["train_games"]
            for condition in args.conditions:
                for seed in ([-1] if condition == "clean" else args.seeds):
                    print("decode", held, condition, seed, flush=True)
                    inputs = inputs_for(held, condition, seed, ball)
                    gt = metrica.labels(held, inputs.n)
                    baseline, dbg = pipeline.predict(inputs, p, debug=True)
                    op = ownership.proba(owner, ownership.features(dbg["trk"], inputs.blocks))
                    sp = supervised_shape.proba(shape, supervised_shape.features(inputs, p))
                    new_q = supervised_shape.replace_team(dbg["shape"], sp)
                    se, pd = fuse.emissions(dbg["ev"], new_q, inputs.tags, dbg["events"], p)
                    np.testing.assert_array_equal(pd, dbg["p_dead"])
                    np.testing.assert_array_equal(se[:, 4], dbg["emit"][:, 4])
                    for variant in manifest["variants"]:
                        emit = se if variant in ("S_shape", "BS_both") else dbg["emit"]
                        bonus = dbg["bonus"]
                        if variant in ("B_ball", "BS_both"):
                            emit, bonus = ownership.apply(emit, bonus, op, p)
                        if variant == "A":
                            lab = baseline
                        else:
                            lab, _ = hsmm.Model(p).decode(emit, bonus, inputs.blocks, inputs.n)
                        alive = np.isin(gt, ["H", "A"])
                        old_home = dbg["shape"][:, 0] / np.maximum(dbg["shape"][:, :2].sum(1), 1e-9)
                        qhome = sp if variant in ("S_shape", "BS_both") else old_home
                        missing = alive & ~dbg["ev"]["seen"]
                        row = {"game": held, "train_games": games, "condition": condition,
                               "seed": seed, "variant": variant,
                               **metrics(lab, gt, dbg["ev"]["seen"], baseline),
                               "shape_brier_alive": float(np.mean((qhome[alive] - (gt[alive] == "H")) ** 2)),
                               "shape_team_acc_missing": float(np.mean((qhome[missing] > 0.5) == (gt[missing] == "H")))}
                        log.write(json.dumps(row) + "\n")
                        log.flush()
                        print(json.dumps(row), flush=True)
                    del inputs, dbg, op, sp, se, new_q
                    gc.collect()


if __name__ == "__main__":
    main()
