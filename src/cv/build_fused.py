"""Fuse the ATD ball with the video detections and write cache/cv/<name>.npz for the pipeline.

    python src/cv/build_fused.py --tag atd --name fused_v1
"""

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
ROOT = os.path.dirname(SRC)
sys.path[:0] = [SRC, ROOT]
from cv import balltrack  # noqa: E402
from games import GAME, GAMES, half_video_blocks  # noqa: E402
from possession import atd as atd_io  # noqa: E402
from starter.load_homography import load as load_homography  # noqa: E402


GRID_MIN_SCORE = 0.3  # ATD-independent grid detections need more confidence than ATD-centred ones


def build(tag="atd", prm=balltrack.Params, blocks=None):
    atd = atd_io.load()
    H = load_homography()[0]
    floors = {t: GRID_MIN_SCORE for t in tag.split("+") if t.startswith("grid")}
    cands = balltrack.fill_skipped(balltrack.load_candidates(tag, floors))
    blocks = blocks or half_video_blocks(GAME)
    return balltrack.fuse(atd["ball"][:, :2], cands, H, blocks, GAMES[GAME]["pitch"], prm,
                          players=atd["players"]), cands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="atd")
    ap.add_argument("--name", default="fused_v1")
    args = ap.parse_args()
    t = time.time()
    out, cands = build(args.tag)
    path = os.path.join(ROOT, "cache", "cv", args.name + ".npz")
    np.savez(path, **out)
    src = out["source"]
    print("wrote {} in {:.0f}s: frames with candidates {}, video {} atd {} none {}".format(
        path, time.time() - t, len(cands), (src == 1).sum(), (src == 2).sum(), (src == 0).sum()))


if __name__ == "__main__":
    main()
