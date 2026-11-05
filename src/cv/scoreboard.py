"""When does the on-screen score change? Pixel change in the score digits of the broadcast
graphic (top-left of the tactical video), sampled once a second inside an interval.

    python src/cv/scoreboard.py 156757 163000
"""

import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), os.path.dirname(os.path.dirname(HERE))]
from games import video_path  # noqa: E402

SCORE_BOX = (260, 14, 312, 40)  # x0, y0, x1, y1 in 1920x1080: the "h-a" digits only, not the clock
STEP = 25


def score_patch(cap, f):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
    ok, im = cap.read()
    if not ok:
        return None
    x0, y0, x1, y1 = SCORE_BOX
    grey = cv2.cvtColor(im[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    return (grey > 190).astype(np.float32) * 100.0  # white digits on a dark translucent box


def change_frame(lo, hi, thresh=3.0, cap=None):
    """First sampled frame in [lo, hi) whose score digits differ from those at lo (and keep
    differing at the next sample), or None if the score never changes."""
    cap = cap or cv2.VideoCapture(video_path())
    ref = score_patch(cap, lo)
    frames = list(range(lo, hi, STEP))
    diffs = []
    for f in frames:
        p = score_patch(cap, f)
        diffs.append(np.abs(p - ref).mean() if p is not None else 0.0)
    diffs = np.array(diffs)
    for i in range(len(frames) - 1):
        if diffs[i] > thresh and diffs[i + 1] > thresh:
            return frames[i], diffs
    return None, diffs


if __name__ == "__main__":
    lo, hi = int(sys.argv[1]), int(sys.argv[2])
    f, d = change_frame(lo, hi)
    print("change at", f, "max diff %.1f" % d.max(), "median %.1f" % np.median(d))
