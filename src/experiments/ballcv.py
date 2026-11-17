"""Building blocks of the CV ball-recovery experiment (step 5). Not used by the pipeline.

proposals  small bright blobs (white top-hat) inside the projected pitch, sized by the homography
features   a normalised grey patch plus colour and context numbers per candidate

cv_probe.py uses these to check whether the ATD ball, projected through the homography, can serve
as a free training label for a ball detector. It cannot, reliably: see cv_probe.py.
"""

import cv2
import numpy as np

PATCH = 12
BALL_M = 0.22
FEATURES = PATCH * PATCH + 16


def project(h, x, y):
    den = h[2, 0] * x + h[2, 1] * y + h[2, 2]
    return (h[0, 0] * x + h[0, 1] * y + h[0, 2]) / den, (h[1, 0] * x + h[1, 1] * y + h[1, 2]) / den


def ball_px(h, x, y, length):
    u0, v0 = project(h, x, y)
    u1, v1 = project(h, x + BALL_M / length, y)
    return np.hypot(u1 - u0, v1 - v0)


def proposals(frame, h, pitch, tophat_px=15, threshold=25, max_area=400, margin=3.0,
              min_ratio=0.05, max_ratio=4.0, max_candidates=150):
    """(K, 9) candidates: u, v, x, y, area, peak, w, h, expected size."""
    length, width = pitch
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tophat_px, tophat_px))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    _, mask = cv2.threshold(tophat, threshold, 255, cv2.THRESH_BINARY)
    n, _, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return np.zeros((0, 9), dtype=np.float32)
    stats, cents = stats[1:], cents[1:]
    area = stats[:, cv2.CC_STAT_AREA]
    keep = (area >= 2) & (area <= max_area)
    stats, cents, area = stats[keep], cents[keep], area[keep]
    if not len(area):
        return np.zeros((0, 9), dtype=np.float32)
    inv = np.linalg.inv(h)
    x, y = project(inv, cents[:, 0], cents[:, 1])
    margin_x, margin_y = margin / length, margin / width
    inside = (x > -margin_x) & (x < 1 + margin_x) & (y > -margin_y) & (y < 1 + margin_y)
    if not inside.any():
        return np.zeros((0, 9), dtype=np.float32)
    stats, cents, area, x, y = stats[inside], cents[inside], area[inside], x[inside], y[inside]
    size = ball_px(h, x, y, length)
    ui = np.clip(cents[:, 0].round().astype(int), 0, gray.shape[1] - 1)
    vi = np.clip(cents[:, 1].round().astype(int), 0, gray.shape[0] - 1)
    peak = tophat[vi, ui].astype(np.float32)
    ratio = area / np.maximum(size ** 2, 1.0)
    plausible = (ratio > min_ratio) & (ratio < max_ratio)
    out = np.stack([cents[:, 0], cents[:, 1], x, y, area, peak,
                    stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT], size], axis=1)
    out = out[plausible]
    if len(out) > max_candidates:
        out = out[np.argsort(-out[:, 5])[:max_candidates]]
    return out.astype(np.float32)


def features(frame, cands, players_px):
    """(K, FEATURES) per candidate: a normalised grey patch plus colour and context numbers."""
    if not len(cands):
        return np.zeros((0, FEATURES), dtype=np.float32)
    pad = 48
    gray = cv2.copyMakeBorder(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), pad, pad, pad, pad,
                              cv2.BORDER_REFLECT).astype(np.float32)
    hsv = cv2.copyMakeBorder(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), pad, pad, pad, pad,
                             cv2.BORDER_REFLECT).astype(np.float32)
    out = np.zeros((len(cands), FEATURES), dtype=np.float32)
    for i, (u, v, _, _, area, peak, bw, bh, size) in enumerate(cands):
        half = int(round(min(pad - 2, max(3.0, 1.5 * size))))
        cu, cv_ = int(round(u)) + pad, int(round(v)) + pad
        box = (slice(cv_ - half, cv_ + half + 1), slice(cu - half, cu + half + 1))
        patch = cv2.resize(gray[box], (PATCH, PATCH), interpolation=cv2.INTER_AREA)
        mu, sd = patch.mean(), patch.std() + 1.0
        out[i, :PATCH * PATCH] = ((patch - mu) / sd).ravel()
        grid = cv2.resize(hsv[box], (9, 9), interpolation=cv2.INTER_AREA)
        centre = grid[3:6, 3:6].reshape(-1, 3).mean(0)
        ring = np.concatenate([grid[0], grid[-1], grid[:, 0], grid[:, -1]]).mean(0)
        if len(players_px):
            d = np.hypot(players_px[:, 0] - u, players_px[:, 1] - v) / max(size, 1.0)
            near, crowd = d.min(), (d < 15).sum()
        else:
            near, crowd = 99.0, 0
        out[i, PATCH * PATCH:] = [area, peak, bw, bh, size, area / max(size ** 2, 1.0),
                                  bw / max(bh, 1.0), centre[1], centre[2], ring[0], ring[1],
                                  ring[2], centre[2] - ring[2], near, crowd, mu]
    return out


def players_pixels(h, players_norm):
    ok = ~np.isnan(players_norm[:, 0])
    if not ok.any():
        return np.zeros((0, 2), dtype=np.float32)
    u, v = project(h, players_norm[ok, 0], players_norm[ok, 1])
    return np.stack([u, v], axis=1)
