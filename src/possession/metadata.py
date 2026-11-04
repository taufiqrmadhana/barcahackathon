"""Everything the ATD XML says about how to read the TXT, as one object.

The XML is a dictionary, not a time series: slot -> team, team names, which team is home, field
size, frame rate, channel units and the first-half attack direction. The half boundaries in the
XML are a single fake half covering the whole recording, so the real halves come from games.py.
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np

from .paths import ROOT  # noqa: F401  (puts the upstream modules on sys.path)
from games import GAME, GAMES, half_video_blocks, require, source_frames

HOME, AWAY = 0, 1
DIRECTIONS = {"left_to_right": 1.0, "right_to_left": -1.0}


@dataclass
class MatchMeta:
    source: str
    n_frames: int
    fps: float
    pitch_xml: tuple             # (length, width) from the XML, metres
    pitch_m: tuple               # real size from games.py, what metres are computed with
    team_ids: tuple              # (home id, away id)
    team_names: tuple            # (home name, away name)
    slot_team: np.ndarray        # (slots,) 0 home / 1 away, slot i is TXT column group i
    channel_units: dict          # channel id -> unit as declared (speed says kph, is m/s)
    home_attack_first: float     # +1 home attacks towards x=1 in the first half, -1 otherwise
    blocks: list = field(default_factory=list)  # [(half, first_frame, last_frame)] video frames

    def home_attack_sign(self, half):
        """+1 if the home team attacks towards x = 1 in that half. Teams swap at halftime."""
        return self.home_attack_first if half == 1 else -self.home_attack_first

    def attack_sign_per_frame(self):
        """(n_frames,) home attack sign per frame, 0 outside the halves."""
        out = np.zeros(self.n_frames, dtype=np.float32)
        for half, first, last in self.blocks:
            out[first:last + 1] = self.home_attack_sign(half)
        return out

    def half_mask(self):
        out = np.zeros(self.n_frames, dtype=bool)
        for _, first, last in self.blocks:
            out[first:last + 1] = True
        return out


def _param(node, name):
    for p in node.iter("ProviderParameter"):
        if (p.findtext("Name") or "").strip() == name:
            return (p.findtext("Value") or "").strip()
    return None


def load(game=GAME, data_dir=None, source="tactical"):
    meta_path = require(["atd_meta"], game, data_dir, source)[0]
    root = ET.parse(meta_path).getroot()

    fps = float(root.findtext(".//FrameRate") or GAMES[game]["fps"])
    length = float(root.findtext(".//FieldSize/Width") or 105)
    width = float(root.findtext(".//FieldSize/Height") or 68)

    teams = list(root.iter("Team"))
    if len(teams) != 2:
        raise ValueError("{}: expected 2 teams, found {}".format(meta_path, len(teams)))
    score = root.find(".//Score")
    home_id = score.get("idLocalTeam") if score is not None else teams[0].get("id")
    teams.sort(key=lambda t: t.get("id") != home_id)  # home first
    team_ids = tuple(t.get("id") for t in teams)
    team_names = tuple(t.findtext("Name") for t in teams)

    direction = _param(teams[0], "attack_direction_first_half") or "left_to_right"
    if direction not in DIRECTIONS:
        raise ValueError("unknown attack direction {!r}".format(direction))

    players = sorted(root.iter("Player"), key=lambda p: int(p.get("id")))
    slot_team = np.array([HOME if p.get("teamId") == team_ids[0] else AWAY for p in players],
                         dtype=np.int8)

    units = {}
    for ch in root.iter("Channel"):
        units[ch.get("id")] = (ch.findtext("Unit") or "").strip()

    return MatchMeta(
        source=source,
        n_frames=source_frames(source, game),
        fps=fps,
        pitch_xml=(length, width),
        pitch_m=tuple(GAMES[game]["pitch"]),
        team_ids=team_ids,
        team_names=team_names,
        slot_team=slot_team,
        channel_units=units,
        home_attack_first=DIRECTIONS[direction],
        blocks=half_video_blocks(game) if source == "tactical" else [],
    )
