"""Stretches that are dead ball by the laws of the game or by the published timeline.

pre-kickoff  graded frames before a half's kickoff (FAQ.md: the second half is graded from 110,808
             but kicks off at 113,742)
after goal   a goal is followed by a kick-off. Smart Tagging marks both (GOALS, then SET PIECES
             "Kick Off (After Goal)"), but its GOALS interval starts with the attack, not the goal.
             The on-screen score changes only once the goal has stood, so from that change to the
             tagged kick-off the ball is certainly dead. A tagged goal after which the score never
             changes (a disallowed goal) adds nothing.

The score change is read from the video (cv/scoreboard.py) once and cached.
"""

import json
import os

from .paths import CACHE_DIR

PRE_KICKOFF = {"BAR-ATH": [(110808, 113742)]}
MAX_GOAL_TO_KICKOFF = 10 * 60 * 25


def goal_intervals(game, instances):
    path = os.path.join(CACHE_DIR, "cv", "goal_dead_{}.json".format(game))
    if os.path.isfile(path):
        with open(path) as handle:
            return [tuple(x) for x in json.load(handle)]
    goals = sorted(i["start_frame"] for i in instances if i["code"].upper() == "GOALS")
    kickoffs = sorted(i["start_frame"] for i in instances
                      if i["tags"].get("Type", "").lower() == "kick off (after goal)")
    from cv.scoreboard import change_frame
    out = []
    for g in goals:
        nxt = [k for k in kickoffs if g < k <= g + MAX_GOAL_TO_KICKOFF]
        if not nxt:
            continue
        k = nxt[0]
        change, _ = change_frame(g, k)
        if change is not None:
            out.append((int(change), int(k)))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(out, handle)
    return out


def dead_intervals(game, p):
    out = []
    if getattr(p, "pre_kickoff_dead", 1):
        out += PRE_KICKOFF.get(game, [])
    if getattr(p, "goal_dead", 1):
        from starter.load_smart_tagging import load
        out += goal_intervals(game, load(game))
    return out
