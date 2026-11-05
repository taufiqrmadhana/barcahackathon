"""Leak-free hidden-ball test of the video ball, on the public window.

For each transplanted mask (the real missing-ball pattern of another five minutes, laid over the
public window -- see robustness.py), the ATD ball is hidden, crop centres are recomputed from the
masked ATD alone, the detector is re-run on every frame whose centre moved, and the fused track is
rebuilt from the masked ATD plus those detections. Writes cache/cv/masked/<seed>.npz with the
masked ATD ball and the fused ball, so the possession pipeline can be run on both.

    python src/cv/masked_eval.py --seeds 0 1 2
"""

import argparse
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
ROOT = os.path.dirname(SRC)
sys.path[:0] = [SRC, ROOT]
from cv import balltrack  # noqa: E402
from cv.detector import Detector  # noqa: E402
from cv.detect_crops import CHUNK  # noqa: E402
from games import GAME, GAMES, video_path, half_video_blocks  # noqa: E402
from possession import atd as atd_io  # noqa: E402
from starter.load_homography import load as load_homography  # noqa: E402

PUBLIC = (11185, 18685)


def transplant_mask(ball, graded, rng, lo=PUBLIC[0], hi=PUBLIC[1]):
    n = hi - lo
    g = graded.copy()
    g[lo - n:hi] = False
    starts = [s for s in np.flatnonzero(g) if g[s:s + n].all()]
    src = int(rng.choice(starts[::25]))
    return np.isnan(ball[src:src + n, 0]), src


def centres_from(ball, H, blocks):
    n = len(ball)
    xy = np.full((n, 2), np.nan)
    for _, first, last in blocks:
        seg = ball[first:last + 1, :2].astype(np.float64)
        idx = np.flatnonzero(~np.isnan(seg[:, 0]))
        t = np.arange(len(seg))
        for k in range(2):
            xy[first:last + 1, k] = np.interp(t, idx, seg[idx, k])
    out = np.full((n, 2), np.nan)
    ok = ~np.isnan(xy[:, 0])
    num = np.einsum("nij,nj->ni", H[ok], np.column_stack([xy[ok], np.ones(ok.sum())]))
    out[ok] = num[:, :2] / num[:, 2:3]
    return out


def base_centres():
    """Crop centres the unmasked pass used, from its chunk files."""
    out = {}
    import glob
    for path in sorted(glob.glob(os.path.join(balltrack.CROPS, "atd", "chunk_*.npz"))):
        d = np.load(path)
        c0 = int(os.path.basename(path)[6:12])
        lo = max(c0, PUBLIC[0]) if c0 < PUBLIC[0] else c0
        cen = d["centre"]
        for i in range(len(cen)):
            out[lo + i] = cen[i]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--move_px", type=float, default=40.0)
    args = ap.parse_args()

    atd = atd_io.load()
    H = load_homography()[0]
    blocks = half_video_blocks(GAME)
    graded = np.zeros(len(atd["ball"]), bool)
    for _, a, b in blocks:
        graded[a:b + 1] = True
    base = balltrack.load_candidates("atd")
    old = base_centres()
    out_dir = os.path.join(ROOT, "cache", "cv", "masked")
    os.makedirs(out_dir, exist_ok=True)
    det = Detector("s_640_int8", 640, people=False)
    for seed in args.seeds:
        path = os.path.join(out_dir, "seed{}.npz".format(seed))
        if os.path.isfile(path):
            print("exists", path)
            continue
        rng = np.random.default_rng(seed)
        mask, src = transplant_mask(atd["ball"], graded, rng)
        ball = atd["ball"].copy()
        hidden = np.zeros(len(ball), bool)
        hidden[PUBLIC[0]:PUBLIC[1]] = mask & ~np.isnan(ball[PUBLIC[0]:PUBLIC[1], 0])
        ball[PUBLIC[0]:PUBLIC[1]][mask] = np.nan
        cen = centres_from(ball, H, blocks)
        redo = [f for f in range(*PUBLIC)
                if f not in old or np.hypot(*(cen[f] - old[f])) > args.move_px]
        cands = dict(base)
        cap = cv2.VideoCapture(video_path())
        redo_set = set(redo)
        cap.set(cv2.CAP_PROP_POS_FRAMES, PUBLIC[0])
        for f in range(*PUBLIC):
            if f not in redo_set:
                cap.grab()
                continue
            ok, image = cap.read()
            det.submit_crop(f, image, float(np.clip(cen[f, 0], 0, 1919)),
                            float(np.clip(cen[f, 1], 0, 1079)), 240)
        for f, (balls, _) in det.drain().items():
            cands[f] = balls
        fused = balltrack.fuse(ball[:, :2], cands, H, blocks, GAMES[GAME]["pitch"],
                                players=atd["players"])
        np.savez(path, atd_ball=ball, fused=fused["xy"], source=fused["source"], hidden=hidden,
                 src=src, redone=np.array(redo))
        pub = slice(*PUBLIC)
        print("seed {} src {}: hidden {} ({:.0%} missing), redone {}, fused video on hidden {:.2f}"
              .format(seed, src, hidden.sum(), np.isnan(ball[pub, 0]).mean(), len(redo),
                      (fused["source"][hidden] == 1).mean()), flush=True)


if __name__ == "__main__":
    main()
