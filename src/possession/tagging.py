"""Smart Tagging as per-frame priors: which team is attacking, and where restarts happen."""

import numpy as np

from .paths import ROOT  # noqa: F401  (puts the upstream modules on sys.path)
from starter.load_smart_tagging import load as load_events
from games import GAME, source_frames

# Tag value -> side. Anything else that is not N/A is an error, not a silent zero.
TEAM_SIDES = {"barcelona": 1.0, "atletic club": -1.0, "athletic club": -1.0}
NO_TEAM = {"", "n/a", "na", "none"}

# Phases where the tagged team has the ball, and phases where the tagged team is defending.
ATTACKING = {
    "build up", "progression", "creating chances", "attacking transition",
    "ball in final third", "ball in the box", "players in the box", "long ball",
}
DEFENDING = {
    "defending in attacking third", "defending in middle third",
    "defending in defensive third", "defensive transition",
}
SET_PIECE = "set pieces"


def _side(event):
    team = (event["tags"].get("Team") or "").strip().lower()
    if team in NO_TEAM:
        return 0.0
    if team not in TEAM_SIDES:
        raise ValueError("unknown Smart Tagging team {!r}".format(team))
    return TEAM_SIDES[team]


def priors(game=GAME, data_dir=None):
    """dict of per-frame arrays.

    team:       sum of +1 (home has it) / -1 (away has it) over the phases covering the frame
    set_piece:  1 inside a tagged set piece, whichever team
    sp_team:    side of the restarting team inside set pieces, 0 when untagged (throw-ins)
    sp_start:   frames where a set piece interval starts
    """
    n = source_frames("tactical", game)
    team = np.zeros(n, dtype=np.float32)
    set_piece = np.zeros(n, dtype=np.float32)
    sp_team = np.zeros(n, dtype=np.float32)
    sp_start = []

    for event in load_events(game, data_dir):
        code = (event["code"] or "").strip().lower()
        start, stop = max(0, event["start_frame"]), min(n, event["end_frame"])
        if stop <= start:
            continue
        side = _side(event)
        if code in ATTACKING:
            team[start:stop] += side
        elif code in DEFENDING:
            team[start:stop] -= side
        elif code == SET_PIECE:
            set_piece[start:stop] = 1.0
            sp_team[start:stop] = side
            sp_start.append(start)

    return {"team": team, "set_piece": set_piece, "sp_team": sp_team,
            "sp_start": np.array(sorted(sp_start), dtype=np.int64)}
