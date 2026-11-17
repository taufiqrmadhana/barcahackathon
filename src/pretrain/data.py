"""External matches -> one common format: players and labels on the competition's 25 fps clock.

A match is a dict of arrays, saved as .npz:

    xy      (N, S, 2) float32  player positions in metres, NaN when off the pitch / not tracked
    teams   (S,)      int8     0 home, 1 away (slot -> team)
    label   (N,)      int8     0 H, 1 A, 2 D, -1 unknown
    blocks  (B, 3)    int64    (half id, first frame, last frame) of every half, inclusive
    pitch   (2,)      float32  length, width in metres

Labels come straight from the provider's own frame flags (kloppy's ball_state and
ball_owning_team), so any provider that fills them works. Event-only providers do not.

    python src/pretrain/data.py sportec J03WPY J03WN1 ...   # downloads open DFL/IDSSE matches
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from possession.paths import ROOT  # noqa: E402

FPS = 25
EXT_DIR = os.environ.get("BARCA_EXT_DIR") or os.path.join(ROOT, "pretrain_data")
MATCH_DIR = os.path.join(EXT_DIR, "matches")

# The seven matches of the open DFL / IDSSE release (Bundesliga and 2. Bundesliga, CC BY 4.0).
SPORTEC_OPEN = ("J03WMX", "J03WN1", "J03WPY", "J03WOH", "J03WQQ", "J03WR9", "J03WOY")


def from_kloppy(ds):
    """kloppy TrackingDataset -> match dict (frames outside the halves are dropped)."""
    meta = ds.metadata
    dims = meta.pitch_dimensions
    length = float(getattr(dims, "pitch_length", None) or 105.0)
    width = float(getattr(dims, "pitch_width", None) or 68.0)
    home, away = meta.teams[0], meta.teams[1]

    slot, teams = {}, []
    for side, team in enumerate((home, away)):
        for player in team.players:
            slot[player.player_id] = len(teams)
            teams.append(side)

    frames = [f for f in ds.frames if f.period is not None]
    n = len(frames)
    xy = np.full((n, len(teams), 2), np.nan, dtype=np.float32)
    label = np.full(n, -1, dtype=np.int8)
    period = np.zeros(n, dtype=np.int8)
    scale = np.array([length, width], dtype=np.float32)
    for i, f in enumerate(frames):
        period[i] = f.period.id
        for player, data in f.players_data.items():
            c, j = data.coordinates, slot.get(player.player_id)
            if c is not None and j is not None:
                xy[i, j] = (c.x, c.y)
        state = str(f.ball_state)
        if state.endswith("DEAD"):
            label[i] = 2
        elif state.endswith("ALIVE") and f.ball_owning_team is not None:
            owner = f.ball_owning_team.team_id
            label[i] = 0 if owner == home.team_id else 1 if owner == away.team_id else -1
    xy *= scale

    blocks = []
    for pid in np.unique(period):
        idx = np.flatnonzero(period == pid)
        blocks.append((int(pid), int(idx[0]), int(idx[-1])))
    return {"xy": xy, "teams": np.array(teams, dtype=np.int8), "label": label,
            "blocks": np.array(blocks, dtype=np.int64),
            "pitch": np.array([length, width], dtype=np.float32)}, float(meta.frame_rate)


def resample(match, fps_in, fps_out=FPS):
    """Linear resample each half to fps_out (positions NaN if either neighbour is NaN)."""
    if abs(fps_in - fps_out) < 1e-6:
        return match
    xy_out, lab_out, blocks = [], [], []
    cursor = 0
    for pid, first, last in match["blocks"]:
        src = match["xy"][first:last + 1]
        lab = match["label"][first:last + 1]
        m = int((len(src) - 1) / fps_in * fps_out) + 1
        t = np.arange(m) / fps_out * fps_in
        lo = np.minimum(np.floor(t).astype(int), len(src) - 1)
        hi = np.minimum(lo + 1, len(src) - 1)
        w = (t - lo).astype(np.float32)[:, None, None]
        xy_out.append((1 - w) * src[lo] + w * src[hi])
        lab_out.append(lab[np.rint(t).astype(int).clip(0, len(src) - 1)])
        blocks.append((pid, cursor, cursor + m - 1))
        cursor += m
    return {**match, "xy": np.concatenate(xy_out), "label": np.concatenate(lab_out),
            "blocks": np.array(blocks, dtype=np.int64)}


def save(match, name):
    os.makedirs(MATCH_DIR, exist_ok=True)
    path = os.path.join(MATCH_DIR, name + ".npz")
    np.savez_compressed(path, **match)
    return path


def stored_matches():
    """Names of the real matches on disk (the synthetic ones are for plumbing tests only)."""
    if not os.path.isdir(MATCH_DIR):
        return []
    return [os.path.splitext(f)[0] for f in sorted(os.listdir(MATCH_DIR))
            if f.endswith(".npz") and not f.startswith("synthetic")]


def load(name_or_path):
    path = name_or_path if os.path.isfile(name_or_path) else os.path.join(MATCH_DIR, name_or_path + ".npz")
    with np.load(path) as stored:
        return {k: stored[k] for k in stored.files}


def summary(match):
    lab = match["label"]
    graded = lab >= 0
    return "{} frames, {} slots, H {:.2f} A {:.2f} D {:.2f}, unlabelled {:.2f}".format(
        len(lab), len(match["teams"]), *[(lab == c).sum() / max(1, graded.sum()) for c in (0, 1, 2)],
        1 - graded.mean())


def download_sportec(match_id):
    from kloppy import sportec
    ds = sportec.load_open_tracking_data(match_id=match_id)
    match, fps = from_kloppy(ds)
    return resample(match, fps)


def main():
    if len(sys.argv) < 2 or sys.argv[1] != "sportec":
        print(__doc__)
        return
    for match_id in sys.argv[2:] or SPORTEC_OPEN:
        path = os.path.join(MATCH_DIR, "sportec_" + match_id + ".npz")
        if os.path.isfile(path):
            print(match_id, "already there")
            continue
        try:
            match = download_sportec(match_id)
        except Exception as error:  # a wrong match id or a dropped connection must not stop the rest
            print(match_id, "FAILED:", type(error).__name__, str(error)[:200], flush=True)
            continue
        print(match_id, save(match, "sportec_" + match_id), summary(match), flush=True)


if __name__ == "__main__":
    main()
