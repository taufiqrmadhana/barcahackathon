"""Pick the frames for the grid pass: graded frames where pass 1 gave no usable video ball and
either the ATD has no ball or its ball is a likely ghost (no player within GHOST_M), restricted to
frames a reference prediction calls alive (+- PAD frames), at a stride.

    python src/cv/select_grid.py --pred evaluation/results/ownership_v1.csv --stride 2
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
ROOT = os.path.dirname(SRC)
sys.path[:0] = [SRC, ROOT]
from games import GAME, half_video_blocks  # noqa: E402
from possession import atd as atd_io  # noqa: E402

GHOST_M = 6.0
PAD = 50


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=os.path.join(ROOT, "evaluation/results/ownership_v1.csv"))
    ap.add_argument("--fused", default=os.path.join(ROOT, "cache/cv/fused_atd.npz"))
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--out", default=os.path.join(ROOT, "cache/cv/grid_frames.npy"))
    ap.add_argument("--exclude", default=os.path.join(ROOT, "cache/cv/grid_frames_pub.npy"))
    ap.add_argument("--public", action="store_true",
                    help="write the public-window lists instead: grid_frames_pub.npy (frames where "
                         "pass 1 gave no video ball) and grid_frames_pub_rest.npy (the others)")
    args = ap.parse_args()
    if args.public:
        src = np.load(args.fused)["source"]
        frames = np.arange(11185, 18685)
        pub = frames[src[frames] != 1]
        np.save(os.path.join(ROOT, "cache/cv/grid_frames_pub.npy"), pub)
        np.save(os.path.join(ROOT, "cache/cv/grid_frames_pub_rest.npy"), np.setdiff1d(frames, pub))
        print("public window: {} + {} frames".format(len(pub), 7500 - len(pub)))
        return
    atd = atd_io.load()
    ball, P = atd["ball"], atd["players"]
    L, W = atd["pitch"]
    n = len(ball)
    graded = np.zeros(n, bool)
    for _, a, b in half_video_blocks(GAME):
        graded[a:b + 1] = True
    src = np.load(args.fused)["source"]
    with np.errstate(all="ignore"):
        d = np.nanmin(np.hypot(P[:, :, 0] * L - ball[:, None, 0] * L,
                               P[:, :, 1] * W - ball[:, None, 1] * W), axis=1)
    ghost = ~np.isnan(ball[:, 0]) & ~(d < GHOST_M)
    need = graded & (src != 1) & (np.isnan(ball[:, 0]) | ghost)
    lab = pd.read_csv(args.pred)["label"].values
    alive = np.isin(lab, ["H", "A"]).astype(np.int8)
    alive = np.convolve(alive, np.ones(2 * PAD + 1, np.int8), mode="same") > 0
    pick = np.flatnonzero(need & alive)
    pick = pick[pick % args.stride == 0]
    if args.exclude and os.path.isfile(args.exclude):
        pick = np.setdiff1d(pick, np.load(args.exclude))
    np.save(args.out, pick)
    print("need {} (missing {}, ghost {}), alive-ish {}, picked {} at stride {}".format(
        need.sum(), (need & np.isnan(ball[:, 0])).sum(), (need & ghost).sum(),
        (need & alive).sum(), len(pick), args.stride))


if __name__ == "__main__":
    main()
