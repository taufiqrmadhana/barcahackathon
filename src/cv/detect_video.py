"""Run the ball detector over a frame range of the tactical video and store the candidates.

    python src/cv/detect_video.py --start 11185 --end 18685 --stride 1
    python src/cv/detect_video.py --start 11185 --end 190198 --stride 2 --offset 0

Resumable: results go to cache/cv/det/<tag>/chunk_<first>.npz, one file per CHUNK frames, and a
chunk already on disk is skipped. Frames are read sequentially (one seek per chunk), so frame i of
the output is frame i of the video, the same index as a submission row.
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
from games import video_path  # noqa: E402

CHUNK = 1000


def pack(results):
    frames = np.array(sorted(results), dtype=np.int64)
    balls = [results[f][0] for f in frames]
    people = [results[f][1] for f in frames]
    nb = np.array([len(b) for b in balls], dtype=np.int32)
    npp = np.array([len(p) for p in people], dtype=np.int32)
    return {
        "frames": frames,
        "n_ball": nb,
        "ball": np.concatenate(balls) if nb.sum() else np.zeros((0, 5), np.float32),
        "n_people": npp,
        "people": np.concatenate(people) if npp.sum() else np.zeros((0, 8), np.float32),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--end", type=int, required=True, help="exclusive")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--model", default="s_1280_int8")
    ap.add_argument("--size", default="1280x736")
    args = ap.parse_args()

    w, h = (int(v) for v in args.size.split("x"))
    tag = "{}_s{}o{}".format(args.model, args.stride, args.offset % args.stride)
    out_dir = os.path.join(ROOT, "cache", "cv", "det", tag)
    os.makedirs(out_dir, exist_ok=True)
    det = Detector(os.path.join(ROOT, "cache", "cv", args.model), w, h, people=True)
    cap = cv2.VideoCapture(video_path())

    first_chunk = (args.start // CHUNK) * CHUNK
    for c0 in range(first_chunk, args.end, CHUNK):
        path = os.path.join(out_dir, "chunk_{:06d}.npz".format(c0))
        if os.path.isfile(path):
            continue
        lo, hi = max(c0, args.start), min(c0 + CHUNK, args.end)
        t = time.time()
        cap.set(cv2.CAP_PROP_POS_FRAMES, lo)
        n = 0
        for f in range(lo, hi):
            if (f - args.offset) % args.stride:
                if not cap.grab():
                    break
                continue
            ok, image = cap.read()
            if not ok:
                break
            det.submit(f, image)
            n += 1
        res = det.drain()
        np.savez(path, **pack(res))
        print("{} frames {}-{}: {} in {:.0f}s ({:.1f} fps)".format(tag, lo, hi, n, time.time() - t,
                                                                 n / max(1e-6, time.time() - t)),
              flush=True)


if __name__ == "__main__":
    main()
