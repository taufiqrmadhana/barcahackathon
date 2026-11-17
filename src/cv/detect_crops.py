"""Pass 1 of the video ball detector: one native-resolution crop per frame around where the ATD
says the ball is (linearly interpolated across ATD gaps, in pitch coordinates), run through the
COCO detector. Resumable chunks in cache/cv/crops/<tag>/chunk_<first>.npz.

    python src/cv/detect_crops.py --start 11185 --end 18685
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
ROOT = os.path.dirname(SRC)
sys.path[:0] = [SRC, ROOT]
from cv.detector import Detector  # noqa: E402
from games import GAME, video_path, half_video_blocks  # noqa: E402
from possession import atd as atd_io  # noqa: E402
from starter.load_homography import load as load_homography, project  # noqa: E402

CHUNK = 1000


def atd_centres():
    """Image position per frame of the ATD ball, interpolated across gaps within each half."""
    ball = atd_io.load()["ball"][:, :2].astype(np.float64)
    H = load_homography()[0]
    n = len(ball)
    xy = np.full((n, 2), np.nan)
    for _, first, last in half_video_blocks(GAME):
        seg = ball[first:last + 1]
        idx = np.flatnonzero(~np.isnan(seg[:, 0]))
        t = np.arange(len(seg))
        for k in range(2):
            xy[first:last + 1, k] = np.interp(t, idx, seg[idx, k])
    out = np.full((n, 2), np.nan)
    ok = ~np.isnan(xy[:, 0])
    num = np.einsum("nij,nj->ni", H[ok], np.column_stack([xy[ok], np.ones(ok.sum())]))
    out[ok] = num[:, :2] / num[:, 2:3]
    return out


def pack(results):
    keys = np.array(sorted(results), dtype=np.int64)
    balls = [results[k][0] for k in keys]
    ppl = [results[k][1] for k in keys]
    nb = np.array([len(b) for b in balls], dtype=np.int32)
    npl = np.array([len(p) for p in ppl], dtype=np.int32)
    return {"frames": keys, "n_ball": nb,
            "ball": np.concatenate(balls) if nb.sum() else np.zeros((0, 5), np.float32),
            "n_people": npl,
            "people": np.concatenate(ppl) if npl.sum() else np.zeros((0, 8), np.float32)}


def run(start, end, centres, tag, model="s_640_int8", half=240):
    out_dir = os.path.join(ROOT, "cache", "cv", "crops", tag)
    os.makedirs(out_dir, exist_ok=True)
    det = Detector(model, 640)
    cap = cv2.VideoCapture(video_path())
    for c0 in range((start // CHUNK) * CHUNK, end, CHUNK):
        path = os.path.join(out_dir, "chunk_{:06d}.npz".format(c0))
        if os.path.isfile(path):
            continue
        lo, hi = max(c0, start), min(c0 + CHUNK, end)
        t = time.time()
        cap.set(cv2.CAP_PROP_POS_FRAMES, lo)
        n = 0
        for f in range(lo, hi):
            if np.isnan(centres[f, 0]):
                if not cap.grab():
                    break
                continue
            ok, image = cap.read()
            if not ok:
                break
            cx = float(np.clip(centres[f, 0], 0, 1919))
            cy = float(np.clip(centres[f, 1], 0, 1079))
            det.submit_crop(f, image, cx, cy, half)
            n += 1
        np.savez(path, centre=centres[lo:hi], **pack(det.drain()))
        dt = time.time() - t
        print("{} {}-{}: {} frames {:.0f}s ({:.1f} fps)".format(tag, lo, hi, n, dt, n / max(dt, 1e-6)),
              flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--end", type=int, required=True, help="exclusive")
    ap.add_argument("--tag", default="atd")
    args = ap.parse_args()
    run(args.start, args.end, atd_centres(), args.tag)


if __name__ == "__main__":
    main()
