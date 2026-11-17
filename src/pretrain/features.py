"""Match -> per-half feature sequences, using the very code the shape model uses on the ATD.

Same cleaning, same `shape.features` (175 columns), so a network trained here reads the competition
features unchanged. Sequences are sub-sampled to STRIDE frames (5 Hz) after the 1 s / 3 s smoothing
has been computed at 25 fps.
"""

import os
import sys
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from possession import clean, evidence, shape  # noqa: E402
from possession.config import Params  # noqa: E402
from . import augment, data  # noqa: E402

STRIDE = 5
FEAT_DIR = os.path.join(data.EXT_DIR, "features")


def frame_features(xy, teams, blocks, pitch, p=None):
    """(N, 175) features and their names for tracking in metres."""
    p = p or Params()
    arr = np.concatenate([xy, np.zeros(xy.shape[:2] + (1,), dtype=np.float32)], axis=-1)
    players_xy, players_vel = clean.clean_players(arr, np.ones(2, dtype=np.float32),
                                                  tuple(map(float, pitch)), p)
    trk = {"players_xy": players_xy, "players_vel": players_vel, "teams": teams,
           "pitch": tuple(map(float, pitch)),
           "attack": clean.attack_sign(players_xy, teams, blocks, tuple(map(float, pitch)))}
    motion = evidence.team_motion(players_xy, players_vel, teams)
    ev = {"player_speed_smooth": evidence.smooth(motion["speed"], p.slow_window)}
    return shape.features(trk, ev)


def sequences(X, label, blocks, stride=STRIDE):
    """Split into one (features, labels) pair per half, sub-sampled."""
    return [(X[first:last + 1:stride], label[first:last + 1:stride])
            for _, first, last in blocks]


def build(name, versions=3, seed=0, cp=None, force=False):
    """Cache `versions` corrupted feature sets of a stored match under FEAT_DIR.
    Version 0 is the match as it came, the others are corrupted."""
    os.makedirs(FEAT_DIR, exist_ok=True)
    match = None
    paths = []
    for v in range(versions):
        path = os.path.join(FEAT_DIR, "{}_v{}.npz".format(name, v))
        paths.append(path)
        if os.path.isfile(path) and not force:
            continue
        match = match or data.load(name)
        if v == 0:
            xy, teams, label = match["xy"], match["teams"], match["label"]
        else:
            rng = np.random.default_rng([seed, v, zlib.crc32(name.encode())])
            xy, teams, label = augment.corrupt(match, rng, cp)
        X, _ = frame_features(xy, teams, match["blocks"], match["pitch"])
        out = {}
        for k, (x, y) in enumerate(sequences(X, label, match["blocks"])):
            out["x{}".format(k)] = x.astype(np.float32)
            out["y{}".format(k)] = y.astype(np.int8)
        out["tracked"] = np.float32(augment.tracked_per_frame(xy))
        np.savez_compressed(path, **out)
        print("{} v{}: {} halves, tracked/frame {:.1f}".format(
            name, v, len(match["blocks"]), float(out["tracked"])), flush=True)
    return paths


def load(path):
    with np.load(path) as stored:
        halves = sorted(int(k[1:]) for k in stored.files if k.startswith("x"))
        return [(stored["x{}".format(k)], stored["y{}".format(k)]) for k in halves]


def main():
    names = [a for a in sys.argv[1:] if not a.startswith("-")] or data.stored_matches()
    for name in names:
        build(name)


if __name__ == "__main__":
    main()
