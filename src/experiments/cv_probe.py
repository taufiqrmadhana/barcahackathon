"""Can the ATD ball be used to train a ball detector for the frames where it is missing?

Step 5 of the plan was local CV to find the ball where the tracking lost it (41% of graded
frames). The cheapest detector is self-supervised: project the ATD ball into the tactical video
through the homography and use that pixel as a free label. This script checks whether that label
is good enough. It prints aggregate numbers only and never writes or shows a frame.

    python src/experiments/cv_probe.py --frames 150

Checks, all on random graded frames where the ATD has the ball:

  axes        a bright blob (white top-hat >= 40) within 4 px of the projected ball, for the four
              axis conventions. The right convention should stand out.
  offset      the same hit rate with the ATD shifted by a few frames against the video. A peak at
              0 confirms the frame alignment.
  motion      camera-compensated frame difference (previous frame warped by H_t H_{t-3}^-1) at the
              projected ball, split by ball speed. A moving ball should light up.
  proposals   share of frames where the top-hat proposal stage puts a candidate on the ball.
  learnable   a gradient-boosted patch classifier, ball patch vs 30 nearby patches per frame,
              trained on 70% of the frames and scored on the rest.

What it found on BAR-ATH (150 / 500 frames): the x,y convention and the zero offset check out
(bright blob at the ball 0.45 at offset 0, 0.03-0.05 for flipped axes), but more than half of
the projected ATD balls sit on no visible bright blob, fast balls show *less* motion at the
projected point than slow ones (median 4 vs 36: in flight, the ATD ball is an estimate rather
than the pixel detection), proposal recall is 0.23, and the patch classifier reaches AUC 0.65 to
0.72 with the true patch ranked first in about 30% of frames. A detector trained on
these labels would inject wrong ball positions into exactly the frames the decoder cannot check,
so the pipeline does not use it; the shape model covers the missing-ball frames instead.
"""

import argparse
import os
import sys

import cv2
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from possession import atd as atd_io  # noqa: E402
from possession.paths import ROOT  # noqa: E402,F401
from games import half_video_blocks, video_path  # noqa: E402
from starter.load_homography import load as load_homography  # noqa: E402
import ballcv  # noqa: E402


def tophat(frame, size=11):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.morphologyEx(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.MORPH_TOPHAT, kernel)


def bright_at(th, u, v, radius=4, level=40):
    ui, vi = int(round(u)), int(round(v))
    if not (radius <= ui < th.shape[1] - radius and radius <= vi < th.shape[0] - radius):
        return None
    return th[vi - radius:vi + radius + 1, ui - radius:ui + radius + 1].max() >= level


class Video:
    def __init__(self):
        self.cap = cv2.VideoCapture(video_path())

    def read(self, frame):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
        ok, image = self.cap.read()
        if not ok:
            raise IOError("cannot read frame {}".format(frame))
        return image


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--frames", type=int, default=150)
    parser.add_argument("--learn-frames", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    atd = atd_io.load()
    ball, players = atd["ball"], atd["players"]
    length, width = atd["pitch"]
    hom = load_homography()[0]
    graded = np.zeros(len(ball), dtype=bool)
    for _, first, last in half_video_blocks("BAR-ATH"):
        graded[first:last + 1] = True
    seen = np.flatnonzero(graded & ~np.isnan(ball[:, 0]))
    speed = np.zeros(len(ball))
    speed[1:] = np.hypot(*(np.diff(ball[:, :2], axis=0) * [length, width]).T) * 25
    rng = np.random.default_rng(args.seed)
    frames = np.sort(rng.choice(seen[seen > 20], args.frames, replace=False))
    video = Video()

    axes = {"x,y": lambda x, y: (x, y), "x,1-y": lambda x, y: (x, 1 - y),
            "1-x,y": lambda x, y: (1 - x, y), "1-x,1-y": lambda x, y: (1 - x, 1 - y)}
    offsets = list(range(-8, 9, 2))
    hits = {k: [] for k in axes}
    off_hits = {o: [] for o in offsets}
    recall = []
    motion = {"slow": [], "fast": []}
    for f in frames:
        image = video.read(f)
        th = tophat(image)
        for name, fn in axes.items():
            hit = bright_at(th, *ballcv.project(hom[f], *fn(ball[f, 0], ball[f, 1])))
            if hit is not None:
                hits[name].append(hit)
        for o in offsets:
            g = f + o
            if not np.isnan(ball[g, 0]):
                hit = bright_at(th, *ballcv.project(hom[f], ball[g, 0], ball[g, 1]))
                if hit is not None:
                    off_hits[o].append(hit)
        u, v = ballcv.project(hom[f], ball[f, 0], ball[f, 1])
        size = ballcv.ball_px(hom[f], ball[f, 0], ball[f, 1], length)
        cands = ballcv.proposals(image, hom[f], (length, width))
        recall.append(len(cands) > 0 and np.hypot(cands[:, 0] - u, cands[:, 1] - v).min()
                      <= max(3.0, size))
        if 40 <= u < 1880 and 40 <= v < 1040:
            prev = cv2.cvtColor(video.read(f - 3), cv2.COLOR_BGR2GRAY)
            warp = cv2.warpPerspective(prev, hom[f] @ np.linalg.inv(hom[f - 3]), (1920, 1080))
            diff = cv2.GaussianBlur(cv2.absdiff(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), warp),
                                    (3, 3), 0)
            ui, vi = int(round(u)), int(round(v))
            at = diff[vi - 4:vi + 5, ui - 4:ui + 5].max()
            local = np.percentile(diff[vi - 40:vi + 41, ui - 40:ui + 41], 95)
            motion["fast" if speed[f] > 5 else "slow"].append((at, local))

    print("frames {}".format(len(frames)))
    print("axes        bright blob at projected ball: " + "  ".join(
        "{} {:.2f}".format(k, np.mean(v)) for k, v in hits.items()))
    print("offset      " + "  ".join("{:+d} {:.2f}".format(o, np.mean(v))
                                     for o, v in off_hits.items()))
    for k, v in motion.items():
        v = np.array(v)
        print("motion      {} n {}  median at ball {:.1f}  local p95 {:.1f}  ball stands out "
              "{:.2f}".format(k, len(v), np.median(v[:, 0]), np.median(v[:, 1]),
                              np.mean(v[:, 0] > np.maximum(25, v[:, 1]))))
    print("proposals   recall {:.2f}".format(np.mean(recall)))

    learn = np.sort(rng.choice(seen[seen > 20], args.learn_frames, replace=False))
    X, y, group = [], [], []
    for f in learn:
        u, v = ballcv.project(hom[f], ball[f, 0], ball[f, 1])
        if not (10 <= u < 1910 and 10 <= v < 1070):
            continue
        image = video.read(f)
        size = ballcv.ball_px(hom[f], ball[f, 0], ball[f, 1], length)
        pts = [(u, v, 1)]
        for _ in range(30):
            r, a = rng.uniform(2.5 * max(size, 3), 80), rng.uniform(0, 2 * np.pi)
            pts.append((u + r * np.cos(a), v + r * np.sin(a), 0))
        pts = [(a, b, c) for a, b, c in pts if 5 <= a < 1915 and 5 <= b < 1075]
        cands = np.array([[a, b, 0, 0, 0, 0, 0, 0, size] for a, b, _ in pts], dtype=np.float32)
        X.append(ballcv.features(image, cands, ballcv.players_pixels(hom[f], players[f, :, :2])))
        y += [c for _, _, c in pts]
        group += [f] * len(pts)
    X, y, group = np.vstack(X), np.array(y), np.array(group)
    cols = list(range(ballcv.PATCH ** 2)) + [ballcv.PATCH ** 2 + i for i in (7, 8, 9, 10, 11, 12, 15)]
    ids = np.unique(group)
    train = np.isin(group, ids[:int(0.7 * len(ids))])
    model = HistGradientBoostingClassifier(max_iter=200, random_state=0)
    model.fit(X[train][:, cols], y[train])
    score = model.predict_proba(X[~train][:, cols])[:, 1]
    first = []
    for f in np.unique(group[~train]):
        k = group[~train] == f
        first.append((score[k] > score[k][y[~train][k] == 1][0]).sum() == 0)
    print("learnable   AUC {:.3f}  true patch ranked first {:.2f}".format(
        roc_auc_score(y[~train], score), np.mean(first)))


if __name__ == "__main__":
    main()
