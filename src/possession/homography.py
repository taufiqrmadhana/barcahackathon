"""Per-frame camera geometry from the homography CSV, normalised and turned into features.

The matrix maps normalised pitch coordinates (0-1, same as the TXT) to image pixels. Raw rows
differ by an arbitrary scale, so every matrix is first normalised to h22 = 1 and given a
quality weight from `valid` and `reprojection_error`. From there:

    aim          image centre projected back onto the pitch (metres): where the camera looks
    view_width   metres of pitch across the centre row of the image: zoom
    pan_speed    how fast the aim moves (m/s): a camera at rest follows a stopped game
    in_view()    whether a pitch point is inside the image on a given frame

The homography never says who has the ball; it says what the camera could see, which is what
turns "the ball is missing" into either "lost in the crowd of players" or "gone out of shot".
"""

import os

import numpy as np
import pandas as pd

from .paths import CACHE_DIR
from games import GAME, require

MATRIX_COLUMNS = ["h{}{}".format(r, c) for r in range(3) for c in range(3)]
RESOLUTIONS = ((1280, 720), (1920, 1080), (2560, 1440), (3840, 2160))


def _read(game, data_dir, source):
    path = require(["homography"], game, data_dir, source)[0]
    cache = os.path.join(CACHE_DIR, "{}_{}_homography.npz".format(game, source))
    if os.path.isfile(cache) and os.path.getmtime(cache) >= os.path.getmtime(path):
        with np.load(cache) as stored:
            return stored["matrix"], stored["error"], stored["valid"]
    table = pd.read_csv(path)
    order = np.argsort(table["frame"].to_numpy())
    matrix = table[MATRIX_COLUMNS].to_numpy(np.float64)[order].reshape(-1, 3, 3)
    error = table["reprojection_error"].to_numpy(np.float64)[order]
    valid = table["valid"].to_numpy()[order].astype(bool)
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez(cache, matrix=matrix, error=error, valid=valid)
    return matrix, error, valid


def normalise(matrix, valid):
    """Scale every matrix to h22 = 1; invalid or degenerate frames become NaN."""
    out = np.full_like(matrix, np.nan)
    scale = matrix[:, 2, 2]
    ok = valid & np.isfinite(matrix).all(axis=(1, 2)) & (np.abs(scale) > 1e-12)
    out[ok] = matrix[ok] / scale[ok, None, None]
    det = np.full(len(matrix), np.nan)
    det[ok] = np.linalg.det(out[ok])
    ok &= np.abs(np.nan_to_num(det)) > 1e-9
    out[~ok] = np.nan
    return out, ok


def apply(matrix, xy):
    """(N,3,3) x (N,K,2) -> (N,K,2) projected points and (N,K) homogeneous w."""
    ones = np.ones(xy.shape[:-1] + (1,), dtype=np.float64)
    p = np.einsum("nij,nkj->nki", matrix, np.concatenate([xy, ones], axis=-1))
    w = p[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        return p[..., :2] / w[..., None], w


def infer_image_size(matrix, ok, ball_norm):
    """Smallest standard resolution that holds the projected tracked ball on almost every frame."""
    seen = ok & ~np.isnan(ball_norm[:, 0])
    idx = np.flatnonzero(seen)[::10]
    if len(idx) == 0:
        return RESOLUTIONS[1]
    px, w = apply(matrix[idx], ball_norm[idx, None, :].astype(np.float64))
    px = px[:, 0][w[:, 0] > 0]
    u99, v99 = np.nanpercentile(px[:, 0], 99.5), np.nanpercentile(px[:, 1], 99.5)
    for width, height in RESOLUTIONS:
        if u99 <= width * 1.05 and v99 <= height * 1.05:
            return width, height
    return RESOLUTIONS[-1]


def _moving_mean(values, window):
    if window <= 1:
        return values
    valid = np.isfinite(values)
    filled = np.where(valid, values, 0.0)
    kernel = np.ones(window) / window
    num = np.convolve(filled, kernel, mode="same")
    den = np.convolve(valid.astype(np.float64), kernel, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


class Camera:
    def __init__(self, game=GAME, data_dir=None, source="tactical", pitch_m=(105.3, 68.0),
                 fps=25.0, ball_norm=None, err_scale=1.0, pan_window=25):
        raw, self.error, self.valid = _read(game, data_dir, source)
        self.matrix, self.ok = normalise(raw, self.valid)
        self.n = len(raw)
        self.pitch = np.asarray(pitch_m, dtype=np.float64)
        self.fps = fps
        self.inverse = np.full_like(self.matrix, np.nan)
        self.inverse[self.ok] = np.linalg.inv(self.matrix[self.ok])

        # trust in [0, 1]: 0 without a matrix, fading as the fit gets worse
        err = np.nan_to_num(self.error, nan=np.inf)
        self.quality = np.where(self.ok, np.exp(-err / err_scale), 0.0)

        if ball_norm is None:
            self.image_size = RESOLUTIONS[1]
        else:
            self.image_size = infer_image_size(self.matrix, self.ok, ball_norm)
        w, h = self.image_size

        # aim and zoom from back-projected image points
        pixels = np.array([[w / 2, h / 2], [0, h / 2], [w, h / 2]], dtype=np.float64)
        back, wb = apply(self.inverse, np.broadcast_to(pixels, (self.n, 3, 2)).copy())
        back = back * self.pitch
        back[wb <= 0] = np.nan
        self.aim = back[:, 0]
        self.view_width = np.hypot(*(back[:, 2] - back[:, 1]).T)
        step = np.full(self.n, np.nan)
        step[1:] = np.hypot(*np.diff(self.aim, axis=0).T) * fps
        self.pan_speed = _moving_mean(step, pan_window)
        zoom = np.full(self.n, np.nan)
        zoom[1:] = np.abs(np.diff(self.view_width)) * fps
        self.zoom_speed = _moving_mean(zoom, pan_window)

    def to_pixels(self, xy_norm):
        """(N,K,2) normalised pitch points -> (N,K,2) pixels, NaN behind the camera."""
        px, w = apply(self.matrix, xy_norm.astype(np.float64))
        px[w <= 0] = np.nan
        return px

    def in_view(self, xy_norm, margin=0.0):
        """(N,K) True where the point is inside the image (shrunk by margin as a fraction)."""
        px = self.to_pixels(xy_norm)
        w, h = self.image_size
        mu, mv = margin * w, margin * h
        with np.errstate(invalid="ignore"):
            return ((px[..., 0] >= mu) & (px[..., 0] <= w - mu)
                    & (px[..., 1] >= mv) & (px[..., 1] <= h - mv))
