"""Pass 2 of the video ball detector: frames where pass 1 (crops at the ATD ball) found nothing
usable get a grid of crops over the playing area, independent of the ATD. Far side (top band):
four 480 px crops upscaled to the network; near side: three native 640 px crops.

    python src/cv/detect_grid.py --frames cache/cv/grid_frames.npy --tag grid

Resumable chunks of CHUNK target frames in cache/cv/crops/<tag>/. Each frame's candidates from all
crops are merged (NMS in frame pixels).
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
from cv.detector import Detector, MAX_CANDIDATES, NMS_IOU, MIN_SCORE, _nms  # noqa: E402
from games import video_path  # noqa: E402

CHUNK = 500
# (cx, cy, half): far band 480 px crops, near band 640 px crops
GRID = [(240, 390, 240), (720, 390, 240), (1200, 390, 240), (1680, 390, 240),
        (320, 790, 320), (960, 790, 320), (1600, 790, 320)]


def merge(parts):
    boxes = np.concatenate([b for b in parts if len(b)]) if any(len(b) for b in parts) \
        else np.zeros((0, 5), np.float32)
    if not len(boxes):
        return boxes
    idx = _nms(boxes[:, :4], boxes[:, 4], MIN_SCORE, NMS_IOU, MAX_CANDIDATES)
    return boxes[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True, help=".npy of target frame indices")
    ap.add_argument("--tag", default="grid")
    args = ap.parse_args()
    frames = np.unique(np.load(args.frames).astype(np.int64))
    out_dir = os.path.join(ROOT, "cache", "cv", "crops", args.tag)
    os.makedirs(out_dir, exist_ok=True)
    det = Detector("s_640_int8", 640, people=False)
    cap = cv2.VideoCapture(video_path())
    pos = -1
    for c in range(0, len(frames), CHUNK):
        path = os.path.join(out_dir, "chunk_{:06d}.npz".format(c))
        if os.path.isfile(path):
            continue
        t = time.time()
        part = frames[c:c + CHUNK]
        for f in part:
            if f != pos:
                if 0 < f - pos <= 50:
                    for _ in range(f - pos):
                        cap.grab()
                else:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
            ok, image = cap.read()
            pos = f + 1
            if not ok:
                continue
            for k, (cx, cy, half) in enumerate(GRID):
                det.submit_crop((int(f), k), image, cx, cy, half)
        res = det.drain()
        balls = []
        for f in part:
            balls.append(merge([res[(int(f), k)][0] for k in range(len(GRID)) if (int(f), k) in res]))
        nb = np.array([len(b) for b in balls], np.int32)
        np.savez(path, frames=part, n_ball=nb,
                 ball=np.concatenate(balls) if nb.sum() else np.zeros((0, 5), np.float32))
        dt = time.time() - t
        print("{} {}/{}: {} frames {:.0f}s ({:.1f} frames/s)".format(
            args.tag, c + len(part), len(frames), len(part), dt, len(part) / dt), flush=True)


if __name__ == "__main__":
    main()
