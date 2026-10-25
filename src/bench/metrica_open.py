"""An outside benchmark: Metrica Sports' open sample games, run through the whole pipeline.

The public window has 17 possessions, so it cannot tell two decoder settings apart. Metrica's two
open sample games (https://github.com/metrica-sports/sample-data, anonymised, "please acknowledge
the source") have tracking with the ball and a full event log by the same company, which gives
about 300 possessions per game. Nothing from the competition's ground truth is involved.

Labels come from the events, not from corrected tracking like the competition's ground truth, so
the convention is close but not the same:

    team      changes at the start of a PASS, RECOVERY, SET PIECE or SHOT by the other team;
              a BALL LOST or a CHALLENGE alone does not change it (a loose ball stays with the
              last team, as in the challenge rules)
    dead      from the ball crossing a line (BALL OUT end, SHOT ...OUT / ...GOAL end), a foul
              (FAULT RECEIVED), an offside, hand ball or end of half, to the next restart
              (SET PIECE, or a PASS of subtype GOAL KICK); dropped if play goes on in between
    X         before the first event and after the last one of each period

The tracking is turned into what the pipeline reads from the ATD: players as anonymous slots of a
team, normalised x/y, no speed column. In these files the ball is NaN whenever it is out of play,
which makes "no ball" an easy dead-ball cue the ATD does not give. `--gaps atd` lays the real
missing-ball mask of a half of the competition ATD over each half (random offset per seed), so
the ball also disappears during play, as it does in the ATD.

    python src/bench/metrica_open.py --download
    python src/bench/metrica_open.py --gaps none atd --seeds 2 --set w_near=0.75 w_near_switch=10

Data goes to pretrain_data/metrica_open/ (gitignored); only aggregates are printed.
"""

import argparse
import hashlib
import json
import os
import sys
import urllib.request

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from possession.config import Params  # noqa: E402
from possession.paths import ROOT  # noqa: E402
from possession import atd as atd_io, pipeline  # noqa: E402
from games import half_video_blocks  # noqa: E402
import possession_metrics as pm  # noqa: E402

URL = "https://raw.githubusercontent.com/metrica-sports/sample-data/master/data/Sample_Game_{g}/{f}"
DATA = os.path.join(ROOT, "pretrain_data", "metrica_open")
FILES = ("RawEventsData", "RawTrackingData_Home_Team", "RawTrackingData_Away_Team")
GAMES = (1, 2)
PITCH = (105.0, 68.0)
POSSESSION = {"PASS", "RECOVERY", "SET PIECE", "SHOT"}
STOP_LOST = ("OFFSIDE", "HAND BALL", "END HALF")
MAX_DEAD = 120 * 25


def path(game, name):
    return os.path.join(DATA, "Sample_Game_{}".format(game),
                        "Sample_Game_{}_{}.csv".format(game, name))


def download():
    for game in GAMES:
        for name in FILES:
            target = path(game, name)
            if os.path.isfile(target):
                continue
            os.makedirs(os.path.dirname(target), exist_ok=True)
            urllib.request.urlretrieve(URL.format(g=game, f=os.path.basename(target)), target)
            print("downloaded", os.path.relpath(target, ROOT))


def tracking(game):
    """(players (N, slots, 3), ball (N, 3), teams, blocks) on the files' own frame index."""
    if game == 3:
        from bench.metrica_epts import load_prepared
        players, ball, teams, blocks = load_prepared()
        return players, ball, teams, [tuple(map(int, block)) for block in blocks]
    sides, ball, periods = [], None, None
    for side, name in enumerate(FILES[1:]):
        table = pd.read_csv(path(game, name), skiprows=2)
        values = table.to_numpy(dtype=np.float64)
        xy = values[:, 3:-2].reshape(len(values), -1, 2)
        sides.append((side, xy))
        ball, periods = values[:, -2:], table["Period"].to_numpy()
    n = len(ball)
    teams = np.concatenate([np.full(xy.shape[1], side, dtype=np.int8) for side, xy in sides])
    players = np.full((n, len(teams), 3), np.nan, dtype=np.float32)
    players[:, :, :2] = np.concatenate([xy for _, xy in sides], axis=1)
    balls = np.full((n, 3), np.nan, dtype=np.float32)
    balls[:, :2] = ball
    blocks = []
    for period in (1, 2):
        idx = np.flatnonzero(periods == period)
        blocks.append((period, int(idx[0]), int(idx[-1])))
    return players, balls, teams, blocks


def events_table(game):
    if game == 3:
        from bench.metrica_epts import event_table
        return event_table()
    return pd.read_csv(path(game, FILES[0]))


def labels(game, n):
    """Per-frame H / A / D / X from the event log (frame index = Start Frame - 1)."""
    return labels_from_events(events_table(game), n)


def labels_from_events(ev, n, possession=POSSESSION):
    """Frozen v1 event convention, also available for explicit sensitivity audits."""
    ev = ev.copy()
    ev["Subtype"] = ev["Subtype"].fillna("")
    ev = ev.sort_values(["Start Frame", "End Frame"], kind="stable").reset_index(drop=True)
    out = np.full(n, "X", dtype="<U1")
    for period in (1, 2):
        e = ev[ev["Period"] == period]
        start = e["Start Frame"].to_numpy() - 1
        end = e["End Frame"].to_numpy() - 1
        kind, sub, team = e["Type"].to_numpy(), e["Subtype"].to_numpy(), e["Team"].to_numpy()
        restart = (kind == "SET PIECE") | ((kind == "PASS") & (np.char.find(
            sub.astype(str), "GOAL KICK") >= 0))
        play = np.isin(kind, ["PASS", "RECOVERY", "SHOT"]) & ~restart

        last = max(start.max(), end.max())
        cur = None
        for i in range(len(e)):
            if kind[i] in possession:
                cur = "H" if team[i] == "Home" else "A"
            stop = start[i + 1] if i + 1 < len(e) else last + 1
            if cur is not None:
                out[start[i]:max(stop, start[i] + 1)] = cur

        for i in range(len(e)):
            s = sub[i]
            if kind[i] == "BALL OUT" or (kind[i] == "SHOT" and ("OUT" in s or "GOAL" in s)):
                dead_from = end[i]
            elif kind[i] == "FAULT RECEIVED" or (kind[i] == "BALL LOST" and any(
                    k in s for k in STOP_LOST)):
                dead_from = start[i]
            else:
                continue
            later = np.flatnonzero((start > dead_from) & restart)
            if not len(later):
                if "END HALF" in s:
                    out[dead_from:last + 1] = "X"
                continue
            dead_to = start[later[0]]
            between = play & (start > dead_from) & (start < dead_to)
            if dead_to - dead_from <= MAX_DEAD and not between.any():
                out[dead_from:dead_to] = "D"
    return out


class BenchInputs(pipeline.Inputs):
    """pipeline.Inputs for an open sample game; no Smart Tagging."""

    def __init__(self, game, gaps=None):  # noqa: super().__init__ is the competition loader
        players, ball, teams, blocks = tracking(game)
        if gaps is not None:
            ball = ball.copy()
            ball[gaps] = np.nan
        # Equal missing-frame counts do not identify equal masks. In particular the
        # pretrained shape cache must never reuse predictions from another mask.
        self.game = "MOS{}".format(game) + ("" if gaps is None else "g{}".format(
            hashlib.sha256(np.asarray(gaps, dtype=bool).tobytes()).hexdigest()[:16]))
        self.atd = {"players": players, "ball": ball, "teams": teams,
                    "names": ("Home", "Away"), "pitch": PITCH}
        self.n = len(ball)
        self.blocks = blocks
        self.graded = np.zeros(self.n, dtype=bool)
        for _, first, last in blocks:
            self.graded[first:last + 1] = True
        self.tags = {"team": np.zeros(self.n, dtype=np.float32),
                     "set_piece": np.zeros(self.n, dtype=np.float32),
                     "sp_team": np.zeros(self.n, dtype=np.float32),
                     "sp_start": np.zeros(0, dtype=np.int64)}
        self._cache = {}


def atd_gaps(n, blocks, rng, competition_ball):
    """The competition ATD's missing-ball mask, one half laid over each half at a random offset."""
    comp = {h: np.isnan(competition_ball[first:last + 1, 0])
            for h, first, last in half_video_blocks("BAR-ATH")}
    mask = np.zeros(n, dtype=bool)
    for half, first, last in blocks:
        length, src = last - first + 1, comp[half]
        if length > len(src):
            src = np.tile(src, int(np.ceil(length / len(src))))
        offset = int(rng.integers(0, len(src) - length + 1))
        mask[first:last + 1] = src[offset:offset + length]
    return mask


def stats(lab, blocks):
    spells = []
    for _, first, last in blocks:
        a = lab[first:last + 1]
        s = 0
        for i in range(1, len(a) + 1):
            if i == len(a) or a[i] != a[s]:
                if a[s] in "HA":
                    spells.append((i - s) / 25.0)
                s = i
    graded = np.concatenate([lab[f:l + 1] for _, f, l in blocks])
    graded = graded[graded != "X"]
    d = np.array(spells)
    minutes = len(graded) / 25.0 / 60.0
    return {"spells": len(d), "per_5min": len(d) / minutes * 5, "mean_s": float(d.mean()),
            "under_4s": float((d < 4).mean()), "dead_share": float((graded == "D").mean())}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--params", default=os.path.join(ROOT, "src", "params.json"))
    parser.add_argument("--set", nargs="+", metavar="KEY=VALUE")
    parser.add_argument("--gaps", nargs="+", default=["none", "atd"], choices=["none", "atd"])
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--json", help="append one line per run to this file")
    parser.add_argument("--describe", action="store_true", help="label statistics only")
    args = parser.parse_args()
    if args.download:
        download()
        return

    from run import parse_overrides
    params = Params.load(args.params, **parse_overrides(args.set))
    competition_ball = None
    for game in GAMES:
        players, ball, teams, blocks = tracking(game)
        gt = labels(game, len(ball))
        if args.describe:
            print("game {}: {}".format(game, {k: round(v, 3) for k, v in stats(gt, blocks).items()}))
            continue
        gt_codes = pm.label_codes(gt)
        for gap in args.gaps:
            for seed in (range(args.seeds) if gap == "atd" else [-1]):
                mask = None
                if gap == "atd":
                    if competition_ball is None:
                        competition_ball = atd_io.load()["ball"]
                    mask = atd_gaps(len(ball), blocks, np.random.default_rng(seed),
                                    competition_ball)
                inputs = BenchInputs(game, mask)
                lab = pipeline.predict(inputs, params)
                r = pm.evaluate(gt_codes, pm.label_codes(lab))
                row = {"game": game, "gaps": gap, "seed": seed, "set": args.set or [],
                       "spellF1": r["spell_f1_mean"], "macroF1": r["macro_f1"],
                       "poss": r["n_pred_spells"], "gt_poss": r["n_gt_spells"],
                       "ball_missing": float(np.isnan(inputs.atd["ball"][inputs.graded, 0]).mean())}
                print("game {game} gaps {gaps:4s} seed {seed:2d}  spellF1 {spellF1:.4f}  macroF1 "
                      "{macroF1:.4f}  poss {poss} (gt {gt_poss})  ball missing {ball_missing:.0%}"
                      .format(**row), flush=True)
                if args.json:
                    with open(args.json, "a") as handle:
                        handle.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
