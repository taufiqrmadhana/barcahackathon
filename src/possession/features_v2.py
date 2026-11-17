"""v2 features: v1 geometry plus homography (camera) cues and per-half normalisation.

Adds, per frame:
    last_ball         last observed ball position in the half (normalised), forward-filled
    since_seen        frames since the ball was last observed (0 when observed)
    last_edge_m       metres from that last position to the nearest touch / goal line
    last_in_view      that last position is still inside the camera image
    players_in_view   tracked players the camera can see
    quality           homography trust, 0-1
    pan_speed, view_width, aim
    z_*               robust z-scores (median / MAD) of the speed signals, per half
"""

import numpy as np

from . import features as v1


def forward_fill(values, mask, blocks):
    """Carry the last row where mask is True forward, restarting at each half."""
    out = np.full_like(values, np.nan)
    since = np.full(len(values), np.iinfo(np.int32).max, dtype=np.int64)
    for _, first, last in blocks:
        seg = slice(first, last + 1)
        m = mask[seg]
        idx = np.where(m, np.arange(m.size), -1)
        idx = np.maximum.accumulate(idx)
        have = idx >= 0
        block = values[seg]
        filled = np.full_like(block, np.nan)
        filled[have] = block[idx[have]]
        out[seg] = filled
        s = np.full(m.size, np.iinfo(np.int32).max, dtype=np.int64)
        s[have] = np.arange(m.size)[have] - idx[have]
        since[seg] = s
    return out, since


def robust_z(values, blocks, n):
    """(x - median) / (1.4826 MAD), statistics computed separately inside each half."""
    out = np.full(n, np.nan, dtype=np.float64)
    for _, first, last in blocks:
        seg = values[first:last + 1].astype(np.float64)
        med = np.nanmedian(seg)
        mad = np.nanmedian(np.abs(seg - med)) * 1.4826
        out[first:last + 1] = (seg - med) / (mad if mad > 1e-9 else 1.0)
    return out


def build(atd, meta, camera, max_gap):
    feat = v1.build(atd, max_gap)
    n = meta.n_frames
    blocks = meta.blocks
    length, width = meta.pitch_m

    ball_norm = atd["ball"][:, :2].astype(np.float64)
    seen = feat["ball_seen"]
    last_ball, since = forward_fill(ball_norm, seen, blocks)
    lx, ly = last_ball[:, 0] * length, last_ball[:, 1] * width
    edge = np.minimum.reduce([lx, length - lx, ly, width - ly])  # negative when outside
    feat["last_ball"] = last_ball
    feat["since_seen"] = since
    feat["last_edge_m"] = edge

    has_last = ~np.isnan(last_ball[:, 0])
    in_view = np.zeros(n, dtype=bool)
    in_view[has_last] = _in_view_rows(camera, last_ball, has_last)
    feat["last_in_view"] = in_view

    players = atd["players"][:, :, :2].astype(np.float64)
    vis = np.zeros(n, dtype=np.int16)
    half = meta.half_mask()
    rows = np.flatnonzero(half)
    for chunk in np.array_split(rows, max(1, len(rows) // 20000)):
        sub = _InViewSlice(camera, chunk)
        v = sub.in_view(np.nan_to_num(players[chunk], nan=-10.0))
        vis[chunk] = (v & ~np.isnan(players[chunk, :, 0])).sum(axis=1)
    feat["players_in_view"] = vis

    feat["quality"] = camera.quality
    feat["pan_speed"] = camera.pan_speed
    feat["view_width"] = camera.view_width
    feat["aim"] = camera.aim
    feat["attack_sign"] = meta.attack_sign_per_frame()

    feat["z_player_speed"] = robust_z(feat["player_speed"], blocks, n)
    feat["z_pan_speed"] = robust_z(camera.pan_speed, blocks, n)
    feat["z_ball_speed"] = robust_z(np.where(seen, feat["ball_speed"], np.nan), blocks, n)
    return feat


class _InViewSlice:
    """Camera.in_view restricted to a subset of frames, so big arrays are processed in chunks."""

    def __init__(self, camera, rows):
        self.camera, self.rows = camera, rows

    def in_view(self, xy_norm, margin=0.0):
        from .homography import apply
        px, w = apply(self.camera.matrix[self.rows], xy_norm)
        px[w <= 0] = np.nan
        width, height = self.camera.image_size
        mu, mv = margin * width, margin * height
        with np.errstate(invalid="ignore"):
            return ((px[..., 0] >= mu) & (px[..., 0] <= width - mu)
                    & (px[..., 1] >= mv) & (px[..., 1] <= height - mv))


def _in_view_rows(camera, points, mask):
    rows = np.flatnonzero(mask)
    return _InViewSlice(camera, rows).in_view(points[rows, None, :], margin=0.03)[:, 0]
