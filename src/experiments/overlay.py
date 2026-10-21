"""Draw the tracking onto the tactical video, for a person to look at locally.

Writes PNGs to cache/overlays/ (gitignored, never committed, never uploaded): a crop around the
projected ATD ball with its trail, every tracked player (home red, away blue), and the top-hat
ball proposals (green), next to the whole frame scaled down. Meant to answer by eye what
cv_probe.py measures in numbers: is the projected ATD ball on the ball, and is the ball visible.
Looking is diagnosis, not labelling: nothing here feeds the pipeline.

    python src/experiments/overlay.py --n 20                 # random frames with the ATD ball
    python src/experiments/overlay.py --fast --n 20          # only frames with a fast ball
    python src/experiments/overlay.py --frames 13710 15350   # specific frames
"""

import argparse
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from possession import atd as atd_io  # noqa: E402
from possession.paths import CACHE_DIR  # noqa: E402
from games import half_video_blocks, video_path  # noqa: E402
from starter.load_homography import load as load_homography  # noqa: E402
import ballcv  # noqa: E402

COLOURS = {0: (40, 40, 230), 1: (230, 120, 30)}


def draw(image, hom, frame, ball, players, teams, pitch, crop=200):
    out = image.copy()
    for s, team in enumerate(teams):
        x, y = players[frame, s, :2]
        if not np.isnan(x):
            u, v = ballcv.project(hom[frame], x, y)
            cv2.circle(out, (int(u), int(v)), 9, COLOURS[int(team)], 2)
    for c in ballcv.proposals(image, hom[frame], pitch):
        cv2.circle(out, (int(c[0]), int(c[1])), 4, (60, 220, 60), 1)
    trail = [f for f in range(frame - 10, frame + 11) if not np.isnan(ball[f, 0])]
    for f in trail:
        u, v = ballcv.project(hom[frame], ball[f, 0], ball[f, 1])
        cv2.circle(out, (int(u), int(v)), 1, (0, 255, 255), -1)
    centre = None
    if not np.isnan(ball[frame, 0]):
        u, v = ballcv.project(hom[frame], ball[frame, 0], ball[frame, 1])
        cv2.circle(out, (int(u), int(v)), 12, (0, 255, 255), 1)
        centre = (int(u), int(v))
    if centre is None:
        centre = (image.shape[1] // 2, image.shape[0] // 2)
    pad = cv2.copyMakeBorder(out, crop, crop, crop, crop, cv2.BORDER_CONSTANT)
    u, v = centre[0] + crop, centre[1] + crop
    zoom = cv2.resize(pad[v - crop:v + crop, u - crop:u + crop], (2 * crop * 2, 2 * crop * 2),
                      interpolation=cv2.INTER_NEAREST)
    small = cv2.resize(out, (zoom.shape[0] * 16 // 9, zoom.shape[0]))
    sheet = np.hstack([zoom, small])
    cv2.putText(sheet, "frame {}".format(frame), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (255, 255, 255), 2)
    return sheet


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--frames", type=int, nargs="*")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=os.path.join(CACHE_DIR, "overlays"))
    args = parser.parse_args()

    atd = atd_io.load()
    ball, players, teams = atd["ball"], atd["players"], atd["teams"]
    length, width = atd["pitch"]
    hom = load_homography()[0]
    frames = args.frames
    if not frames:
        graded = np.zeros(len(ball), dtype=bool)
        for _, first, last in half_video_blocks("BAR-ATH"):
            graded[first:last + 1] = True
        speed = np.zeros(len(ball))
        speed[1:] = np.hypot(*(np.diff(ball[:, :2], axis=0) * [length, width]).T) * 25
        pool = graded & ~np.isnan(ball[:, 0])
        if args.fast:
            pool &= speed > 8
        pool = np.flatnonzero(pool)
        frames = np.sort(np.random.default_rng(args.seed).choice(pool, args.n, replace=False))
    os.makedirs(args.out, exist_ok=True)
    cap = cv2.VideoCapture(video_path())
    for f in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, image = cap.read()
        if not ok:
            continue
        path = os.path.join(args.out, "frame_{:06d}.png".format(int(f)))
        cv2.imwrite(path, draw(image, hom, int(f), ball, players, teams, (length, width)))
    print("wrote {} overlays to {}".format(len(frames), args.out))


if __name__ == "__main__":
    main()
