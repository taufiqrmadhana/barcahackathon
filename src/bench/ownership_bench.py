"""Fixed-config, leave-one-game-out ownership experiment on public Metrica samples.

No model selection on the held-out game; no competition GT is loaded. The competition
ball is read locally only to transplant missingness. Output contains aggregates and
local checkpoints. Run with OMP_NUM_THREADS=4 to avoid tiny GBMs oversubscribing CPUs.
"""

import argparse
import copy
import gc
import hashlib
import json
import os
import sys
from dataclasses import asdict, replace
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench import metrica_open as metrica
from possession import atd as atd_io, fuse, hsmm, ownership, pipeline
from possession.config import Params
from possession.paths import ROOT
import possession_metrics as pm

SEED = 20261026
LABEL_VERSION = "metrica-events-v1"


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def audit(game):
    players, ball, teams, blocks = metrica.tracking(game)
    y = metrica.labels(game, len(ball))
    ev = metrica.events_table(game)
    aligned = True
    for name in (metrica.FILES[1:] if game != 3 else []):
        table = pd.read_csv(metrica.path(game, name), skiprows=2, usecols=["Frame", "Period"])
        aligned &= np.array_equal(table["Frame"], np.arange(len(ball)) + 1)
    if not aligned:
        raise ValueError("Metrica frame index is not contiguous and one-based")
    meaningful = ev[ev.Type.isin(metrica.POSSESSION)]
    conflicts = meaningful.groupby(["Period", "Start Frame"]).Team.nunique()
    missing = ~np.isfinite(ball[:, 0])
    alive, dead = np.isin(y, ["H", "A"]), y == "D"
    if game == 3:
        from bench.metrica_epts import DATA, FILES, source_hashes
        sources = source_hashes()
        aligned = json.loads((DATA / "prepared.json").read_text())["frame_alignment_ok"]
    else:
        sources = {name: sha(metrica.path(game, name)) for name in metrica.FILES}
    carry = metrica.labels_from_events(ev, len(ball), metrica.POSSESSION | {"CARRY"})
    return {"game": game, "label_version": LABEL_VERSION,
            "label_sha256": hashlib.sha256(y.tobytes()).hexdigest(),
            "source_sha256": sources,
            "carry_sensitivity_frames": int((carry != y).sum()),
            **metrica.stats(y, blocks), "frame_alignment_ok": bool(aligned),
            "conflicting_owner_timestamps": int((conflicts > 1).sum()),
            "missing_given_alive": float(missing[alive].mean()),
            "missing_given_dead": float(missing[dead].mean()),
            "notes": ["Labels are event-derived supervision, not corrected possession GT.",
                      "Same-time owner events follow stable Start Frame, End Frame ordering.",
                      "Dead intervals require a restart and no intervening play; advantage is heuristic.",
                      "Independent ATD masking is a stress test, not a full domain match."]}


def corrupt(inputs, seed):
    """Assumed player/ball corruption rates; no labels consulted, no team relabelling.

    Smooth position errors avoid turning independent jitter into unrealistically large
    velocities. Same-team slot exchanges create identity breaks without changing teams.
    """
    rng = np.random.default_rng(seed)
    other = copy.copy(inputs)
    other.atd = dict(inputs.atd)
    players, ball = inputs.atd["players"].copy(), inputs.atd["ball"].copy()
    scale = np.asarray(inputs.atd["pitch"])
    for _, first, last in inputs.blocks:
        count = last - first + 1
        grid = np.arange(count)
        knots = np.arange(0, count + 10, 10)
        for slot in range(players.shape[1]):
            for axis in range(2):
                err = np.interp(grid, knots, rng.normal(0, 0.20, len(knots)))
                players[first:last + 1, slot, axis] += err / scale[axis]
            for _ in range(rng.poisson(count / 25 / 60 * 0.5)):
                a = int(rng.integers(first, last + 1))
                b = min(last + 1, a + int(rng.uniform(1, 8) * 25))
                players[a:b, slot] = np.nan
        for side in (0, 1):
            slots = np.flatnonzero(inputs.atd["teams"] == side)
            for _ in range(rng.poisson(count / 25 / 60)):
                a = int(rng.integers(first, last + 1))
                b = min(last + 1, a + int(rng.uniform(2, 15) * 25))
                left, right = rng.choice(slots, 2, replace=False)
                players[a:b, [left, right]] = players[a:b, [right, left]]
        for axis in range(2):
            err = np.interp(grid, knots, rng.normal(0, 0.10, len(knots)))
            ball[first:last + 1, axis] += err / scale[axis]
    other.atd["players"], other.atd["ball"] = players, ball
    digest = hashlib.sha256(players.tobytes() + ball.tobytes()).hexdigest()[:16]
    other.game = inputs.game + "c" + digest
    other._cache = {}
    return other


def inputs_for(game, condition, seed, competition_ball):
    inputs = metrica.BenchInputs(game)
    if condition != "clean":
        mask = metrica.atd_gaps(inputs.n, inputs.blocks, np.random.default_rng(seed), competition_ball)
        inputs = metrica.BenchInputs(game, mask)
    return corrupt(inputs, seed + 10000) if condition == "corrupt" else inputs


def training_batches(games, p, competition_ball):
    batches = []
    for game in games:
        for condition in ("clean", "corrupt"):
            inputs = inputs_for(game, condition, SEED + game, competition_ball)
            views, support, names = ownership.features(inputs.tracking(p), inputs.blocks)
            labels = metrica.labels(game, inputs.n)
            batches.append((views, support, names, labels, inputs.blocks))
    return batches


def metrics(labels, gt, seen, baseline):
    codes, pred = pm.label_codes(gt), pm.label_codes(labels)
    result = pm.evaluate(codes, pred, bootstrap_n=0)
    runs = pm.valid_runs(codes)
    gs, ps = pm.possession_spells(codes, runs), pm.possession_spells(pred, runs)
    short = np.array([b - a < 100 for a, b, _ in gs])
    overlaps = pm.spell_overlaps(gs, ps)
    short_recall = [float(pm.matched_gt_flags(overlaps, len(gs), threshold)[short].mean())
                    for threshold in pm.SPELL_IOUS] if short.any() else []
    out = {"spellF1": result["spell_f1_mean"], "macroF1": result["macro_f1"],
           "poss": result["n_pred_spells"], "gt_poss": result["n_gt_spells"],
           "short_spell_recall": float(np.mean(short_recall)) if short_recall else None,
           "short_spell_count": int(short.sum()), "alive_dead_f1": result["alive_dead_f1"],
           "dead_frames_changed": int(((labels == "D") != (baseline == "D")).sum())}
    alive = np.isin(gt, ["H", "A"])
    for name, mask in (("seen", seen), ("missing", ~seen)):
        keep = alive & mask
        out["team_acc_" + name] = float((labels[keep] == gt[keep]).mean()) if keep.any() else None
    # Boundaries are matched within each valid run, never across adjacent halves/gaps.
    latencies, matched, n_gt, n_pred = [], 0, 0, 0
    for run in runs:
        g = [t for t in pm.transitions(codes, [run]) if t[1] in (0, 1) and t[2] in (0, 1)]
        pr = [t for t in pm.transitions(pred, [run]) if t[1] in (0, 1) and t[2] in (0, 1)]
        pairs, _, _ = pm.match_transitions(g, pr, tol=25, typed="pair")
        latencies.extend([(b[0] - a[0]) / 25 for a, b in pairs])
        matched += len(pairs)
        n_gt, n_pred = n_gt + len(g), n_pred + len(pr)
    out.update(team_transition_f1_1s=2 * matched / max(1, n_gt + n_pred),
               boundary_median_offset_s=float(np.median(latencies)) if latencies else None,
               boundary_matches=matched)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=os.path.join(ROOT, "cache", "ownership_v1", "logo"))
    ap.add_argument("--conditions", nargs="+", choices=["clean", "atd", "corrupt"],
                    default=["clean", "atd", "corrupt"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--audit-only", action="store_true")
    ap.add_argument("--fit-all", action="store_true", help="Fit deployable checkpoint; no evaluation")
    ap.add_argument("--params", default=os.path.join(ROOT, "src", "configs", "external_training.json"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    p = Params.load(args.params, pre_kickoff_dead=0, goal_dead=0)
    if p.ownership_model or p.w_near or p.w_near_switch:
        raise ValueError("experiment baseline must not already contain ownership/near evidence")
    audits = [audit(g) for g in metrica.GAMES]
    (out / "audit.json").write_text(json.dumps(audits, indent=2) + "\n")
    for item in audits:
        print("audit", item["game"], "spells", item["spells"], "conflicts",
              item["conflicting_owner_timestamps"], flush=True)
    if args.audit_only:
        return
    manifest = {"seed": SEED, "label_version": LABEL_VERSION, "params": asdict(p),
                "conditions": args.conditions, "eval_seeds": args.seeds,
                "training_conditions": ["clean", "corrupt"], "fit_all": args.fit_all,
                "variants": {"A": {}, "A_near": {"w_near": 0.75, "w_near_switch": 10},
                             "B_owner": {"w_owner": 1, "w_owner_switch": 0},
                             "B_both": {"w_owner": 1, "w_owner_switch": 12}},
                "source_hashes": {str(f.relative_to(ROOT)): sha(f)
                    for folder in ("src/possession", "src/bench")
                    for f in sorted(Path(ROOT, folder).glob("*.py"))},
                "data": audits}
    manifest_file = out / "manifest.json"
    if manifest_file.exists() and json.loads(manifest_file.read_text()) != manifest:
        raise ValueError("output directory belongs to a different experiment; use a fresh --out")
    manifest_file.write_text(json.dumps(manifest, indent=2) + "\n")
    competition_ball = atd_io.load()["ball"]
    if args.fit_all:
        batches = training_batches(metrica.GAMES, p, competition_ball)
        artifact = ownership.fit(batches, p, metrica.GAMES)
        artifact["experiment"] = manifest
        joblib.dump(artifact, out / "ownership_all.joblib")
        print("saved", out / "ownership_all.joblib", flush=True)
        return
    with (out / "results.jsonl").open("w") as log:
        for held_out in metrica.GAMES:
            train_games = [g for g in metrica.GAMES if g != held_out]
            assert held_out not in train_games
            print("fit train", train_games, "held out", held_out, flush=True)
            batches = training_batches(train_games, p, competition_ball)
            artifact = ownership.fit(batches, p, train_games)
            artifact["experiment"] = manifest
            joblib.dump(artifact, out / f"ownership_holdout_{held_out}.joblib")
            del batches
            gc.collect()
            for condition in args.conditions:
                for seed in ([-1] if condition == "clean" else args.seeds):
                    inputs = inputs_for(held_out, condition, seed, competition_ball)
                    gt = metrica.labels(held_out, inputs.n)
                    print("decode", held_out, condition, seed, flush=True)
                    baseline, dbg = pipeline.predict(inputs, p, debug=True)
                    predictions = ownership.proba(artifact, ownership.features(dbg["trk"], inputs.blocks))
                    for variant, overrides in manifest["variants"].items():
                        vp = replace(p, **overrides)
                        if variant == "A":
                            lab = baseline
                        elif variant == "A_near":
                            emit, _ = fuse.emissions(dbg["ev"], dbg["shape"], inputs.tags, dbg["events"], vp)
                            bonus = fuse.bonuses(dbg["events"], inputs.n, vp, dbg["ev"])
                            lab, _ = hsmm.Model(vp).decode(emit, bonus, inputs.blocks, inputs.n)
                        else:
                            emit, bonus = ownership.apply(dbg["emit"], dbg["bonus"], predictions, vp)
                            lab, _ = hsmm.Model(vp).decode(emit, bonus, inputs.blocks, inputs.n)
                        row = {"game": held_out, "train_games": train_games, "condition": condition,
                               "seed": seed, "variant": variant,
                               **metrics(lab, gt, dbg["ev"]["seen"], baseline)}
                        log.write(json.dumps(row) + "\n")
                        log.flush()
                        print(json.dumps(row), flush=True)
                    del inputs, dbg, predictions
                    gc.collect()


if __name__ == "__main__":
    main()
