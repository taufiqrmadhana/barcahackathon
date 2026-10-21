"""Load the raw ATD into dense numpy arrays, cached as .npz after the first read.

Raw line format: "<frame>:<slot>;<slot>;...:<ball>", each group "x,y,speed" (NaN when missing).
Every separator is turned into a comma so pandas' C parser reads the whole file in one pass.
"""

import io
import os
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from .paths import CACHE_DIR
from games import GAME, GAMES, require, source_frames

HOME, AWAY = 0, 1


def slot_teams(meta_path):
    """Team of every track slot, 0 home / 1 away. The first team in the metadata is home."""
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError("{}: expected 2 teams, found {}".format(meta_path, len(team_ids)))
    return np.array([HOME if p.get("teamId") == team_ids[0] else AWAY
                     for p in root.iter("Player")], dtype=np.int8)


def team_names(meta_path):
    root = ET.parse(meta_path).getroot()
    return tuple(team.findtext("Name") for team in root.iter("Team"))


def _parse(raw_path, n_slots, n_frames):
    with open(raw_path) as handle:
        text = handle.read().replace(":", ",").replace(";", ",")
    table = pd.read_csv(io.StringIO(text), header=None, dtype=np.float64).to_numpy()
    del text

    n_ball = table.shape[1] - 1 - 3 * n_slots
    if n_ball < 2:
        raise ValueError("unexpected ATD width {} for {} slots".format(table.shape[1], n_slots))

    frames = table[:, 0].astype(np.int64) - 1  # ATD frame_id counts from 1
    keep = (frames >= 0) & (frames < n_frames)
    frames, table = frames[keep], table[keep]

    players = np.full((n_frames, n_slots, 3), np.nan, dtype=np.float32)
    players[frames] = table[:, 1:1 + 3 * n_slots].reshape(-1, n_slots, 3)
    ball = np.full((n_frames, 3), np.nan, dtype=np.float32)
    ball[frames, :min(n_ball, 3)] = table[:, 1 + 3 * n_slots:1 + 3 * n_slots + min(n_ball, 3)]
    return players, ball


def load(game=GAME, data_dir=None, refresh=False):
    """dict with players (N, slots, 3), ball (N, 3), teams (slots,), names, pitch.

    x and y stay normalised 0-1, the third column is speed in m/s.
    """
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    teams = slot_teams(meta_path)
    n_frames = source_frames("tactical", game)
    cache = os.path.join(CACHE_DIR, "{}_tactical_atd.npz".format(game))

    if refresh or not os.path.isfile(cache) or os.path.getmtime(cache) < os.path.getmtime(raw_path):
        players, ball = _parse(raw_path, len(teams), n_frames)
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.savez(cache, players=players, ball=ball)
    else:
        with np.load(cache) as stored:
            players, ball = stored["players"], stored["ball"]

    return {
        "players": players,
        "ball": ball,
        "teams": teams,
        "names": team_names(meta_path),
        "pitch": GAMES[game]["pitch"],
    }
